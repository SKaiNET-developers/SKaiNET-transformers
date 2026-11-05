#!/usr/bin/env python3
"""Optional persistent IREE backend for the SKaiNET LFM2 fixed-capacity graph.

Model parameters and all hybrid state stay in the selected HAL device. Only
token controls and final logits cross the host boundary. This worker implements
no CPU model operators and never falls back to a different HAL device.
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys


PROTOCOL = "skainet.lfm2.iree.v1"
MAX_REQUEST_BYTES = 1024 * 1024


class RequestError(ValueError):
    """Invalid request rejected before changing model state."""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Non-finite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def verified_file(root, record):
    name = record["file"]
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Artifact path must be relative to its bundle")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact must be a regular file inside its bundle")
    expected = record["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Invalid artifact SHA-256")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Artifact digest mismatch: {name}")
    return path


def load_contract(bundle):
    root = Path(bundle).resolve(strict=True)
    manifest = strict_json((root / "runtime-manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported runtime manifest protocol/version")
    if manifest.get("backend") != "vulkan-spirv" or manifest.get("dtype") != "float32":
        raise ValueError("The production bundle must declare Vulkan FP32")
    if manifest.get("embedding_policy") != "one_hot_sharded":
        raise ValueError("Unsupported embedding policy")
    capacity = positive_integer(manifest["capacity"], "capacity")
    dims = manifest["dimensions"]
    vocabulary = positive_integer(dims["vocabulary"], "vocabulary")
    head = positive_integer(dims["head_dimension"], "head_dimension")
    width = positive_integer(dims["width"], "width")
    kv = positive_integer(dims["kv_heads"], "kv_heads")
    kernel = positive_integer(dims["conv_kernel"], "conv_kernel")
    maximum = positive_integer(dims["context_limit"], "context_limit")
    theta = dims["rope_theta"]
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("Invalid rope_theta")
    if head % 2 or capacity > maximum:
        raise ValueError("Invalid rotary geometry or capacity")
    layers = dims["layer_types"]
    if not isinstance(layers, list) or not layers or any(k not in ("conv", "full_attention") for k in layers):
        raise ValueError("Unsupported layer_types")
    states = []
    for index, kind in enumerate(layers):
        if kind == "conv":
            states.append({"name": f"history_{index}", "shape": [width, kernel], "dtype": "f32"})
        else:
            states.extend({"name": f"{kind}_{index}", "shape": [kv, capacity, head], "dtype": "f32"} for kind in ("key", "value"))
    inputs = [
        {"name": "token_one_hot", "shape": [1, vocabulary], "dtype": "f32"},
        {"name": "cos", "shape": [1, head], "dtype": "f32"},
        {"name": "sin", "shape": [1, head], "dtype": "f32"},
        {"name": "mask", "shape": [1, capacity + 1], "dtype": "f32"},
        {"name": "write_mask", "shape": [1, capacity, 1], "dtype": "f32"},
        *states,
    ]
    outputs = [{"name": "logits", "shape": [1, vocabulary], "dtype": "f32"}, *states]
    if manifest["inputs"] != inputs or manifest["outputs"] != outputs:
        raise ValueError("Graph port order/shapes differ from the generation contract")
    if manifest["function_name"] != "lfm2_cached_step" or manifest["parameter_scope"] != "model":
        raise ValueError("Unsupported entrypoint or parameter scope")
    module = verified_file(root, manifest["module"])
    parameters = verified_file(root, manifest["parameters"])
    return manifest, module, parameters


def copy_device_f32_to_host(array, rt, np, *, device):
    """Return owned FP32 host data without IREE 3.11's leaking MappedMemory.asarray.

    The session pins IREE 3.11.0. DeviceArray._buffer_view is the only private
    interop access; allocation, copying, synchronization and mapping use public
    HAL APIs. The caller supplies the owning device explicitly. Device feedback
    never passes through this helper.
    """
    if not isinstance(array, rt.DeviceArray) or array.dtype != np.float32:
        raise ValueError("Expected an FP32 IREE DeviceArray")
    view = array._buffer_view
    if rt.HalElementType.map_to_dtype(view.element_type) != np.dtype(np.float32):
        raise ValueError("Underlying device storage must be FP32")
    shape = tuple(int(dimension) for dimension in array.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError("Negative device array dimension")
    count = math.prod(shape)
    source = view.get_buffer()
    host_visible = int(rt.MemoryType.HOST_VISIBLE)
    scoped_mapping = int(rt.BufferUsage.MAPPING_SCOPED)
    if source.memory_type() & host_visible == host_visible and source.allowed_usage() & scoped_mapping == scoped_mapping:
        mapped = view.map()
    else:
        host_buffer = device.allocator.allocate_buffer(
            memory_type=rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
            allowed_usage=rt.BufferUsage.TRANSFER_TARGET | rt.BufferUsage.MAPPING_SCOPED,
            allocation_size=source.byte_length(),
        )
        semaphore = device.create_semaphore(0)
        device.queue_copy(source, host_buffer,
                          wait_semaphores=rt.HalFence.create_at(semaphore, 0),
                          signal_semaphores=rt.HalFence.create_at(semaphore, 1))
        if not rt.HalFence.create_at(semaphore, 1).wait():
            raise RuntimeError("Device-to-host copy did not complete")
        mapped = host_buffer.map()
    # The temporary NumPy view is copied while its Python buffer export is live.
    # Neither the returned array nor DeviceArray retains the mapping.
    with memoryview(mapped) as memory:
        if memory.nbytes < count * np.dtype(np.float32).itemsize:
            raise ValueError("Mapped buffer is smaller than its declared FP32 shape")
        result = np.frombuffer(memory, dtype=np.float32, count=count).copy()
    return result.reshape(shape)


class IreeSession:
    def __init__(self, bundle, device_uri, *, allow_cpu_for_tests=False):
        self.states = []
        try:
            self._initialize(bundle, device_uri, allow_cpu_for_tests=allow_cpu_for_tests)
        except BaseException:
            self.close()
            raise

    def _initialize(self, bundle, device_uri, *, allow_cpu_for_tests):
        # Import only after the caller's explicit private package directory is set.
        import numpy as np
        import iree.runtime as rt
        self.np, self.rt = np, rt
        self.contract, module_path, parameters_path = load_contract(bundle)
        if importlib.metadata.version("iree-base-runtime") != "3.11.0":
            raise ValueError("This bundle requires the validated IREE runtime 3.11.0")
        if not device_uri.startswith("vulkan://") or not device_uri.removeprefix("vulkan://"):
            if not (allow_cpu_for_tests and device_uri == "local-task"):
                raise ValueError("An explicit Vulkan device URI is required; CPU fallback is disabled")
        self.device_uri = device_uri
        self.device = rt.get_device(device_uri, cache=False)
        config = rt.Config(device=self.device)
        self.parameter_index = rt.ParameterIndex()
        self.parameter_index.load(str(parameters_path))
        self.parameter_provider = self.parameter_index.create_provider(scope="model")
        parameters_module = rt.create_io_parameters_module(config.vm_instance, self.parameter_provider)
        self.module = rt.VmModule.mmap(config.vm_instance, str(module_path))
        self.context = rt.SystemContext(vm_modules=[parameters_module, self.module], config=config)
        self.function = self.context.modules[self.module.name][self.contract["function_name"]]
        self.capacity = self.contract["capacity"]
        self.dims = self.contract["dimensions"]
        self.vocabulary = self.dims["vocabulary"]
        self.state_ports = self.contract["outputs"][1:]
        self.steps = 0
        self.reset()

    def reset(self):
        self.states = [self.rt.asdevicearray(self.device, self.np.zeros(p["shape"], dtype=self.np.float32),
                                           implicit_host_transfer=False) for p in self.state_ports]
        self.position = 0

    def controls(self, token):
        np = self.np
        one_hot = np.zeros((1, self.vocabulary), dtype=np.float32)
        one_hot[0, token] = 1
        head = self.dims["head_dimension"]
        # Match the eager Kotlin control policy: double angle, then FP32 sin/cos.
        angles = [self.position * self.dims["rope_theta"] ** (-2.0 * i / head) for i in range(head // 2)]
        cos = [math.cos(a) for a in angles]
        sin = [math.sin(a) for a in angles]
        cosine = np.array([cos + cos], dtype=np.float32)
        sine = np.array([sin + sin], dtype=np.float32)
        mask = np.full((1, self.capacity + 1), -np.finfo(np.float32).max, dtype=np.float32)
        mask[0, :self.position] = 0
        mask[0, self.capacity] = 0
        write_mask = np.zeros((1, self.capacity, 1), dtype=np.float32)
        write_mask[0, self.position, 0] = 1
        return [one_hot, cosine, sine, mask, write_mask]

    def forward(self, tokens):
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < self.vocabulary for t in tokens):
            raise RequestError("Token IDs must be nonempty integers inside the vocabulary")
        if self.position + len(tokens) > self.capacity:
            raise RequestError("The fixed-capacity context is exhausted")
        logits = None
        for token in tokens:
            controls = [self.rt.asdevicearray(self.device, x, implicit_host_transfer=False) for x in self.controls(token)]
            result = self.function(*controls, *self.states)
            if not isinstance(result, tuple) or len(result) != len(self.contract["outputs"]):
                raise RuntimeError("Compiled function returned a different output contract")
            for port, tensor in zip(self.contract["outputs"], result, strict=True):
                if not isinstance(tensor, self.rt.DeviceArray) or list(tensor.shape) != port["shape"] or tensor.dtype != self.np.float32:
                    raise RuntimeError("Compiled output shape/type differs from the manifest")
                # Device state is passed through unchanged; reject accidental implicit NumPy conversion.
                tensor._implicit_host_transfer = False
            # Return owned host bytes, not a view retaining a mapped HAL allocation.
            logits = copy_device_f32_to_host(result[0], self.rt, self.np, device=self.device)
            if not self.np.isfinite(logits).all():
                raise RuntimeError("GPU forward produced non-finite logits")
            self.states = list(result[1:])
            self.position += 1
            self.steps += 1
        return logits.reshape(self.vocabulary)

    def ready(self):
        return {"protocol": PROTOCOL, "event": "ready", "vocabulary": self.vocabulary,
                "capacity": self.capacity, "position": self.position, "device_uri": self.device_uri,
                "runtime_version": "3.11.0", "state_location": "device", "cpu_model_fallback": False,
                "module_sha256": self.contract["module"]["sha256"],
                "parameters_sha256": self.contract["parameters"]["sha256"]}

    def close(self):
        # Release bound functions before the context, and device arrays before
        # their allocator. The process is owned by one Kotlin runtime instance.
        self.states.clear()
        self.function = None
        self.context = None
        self.module = None
        self.parameter_provider = None
        self.parameter_index = None
        self.device = None
        gc.collect()


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def probe(device_uri):
    import iree.runtime as rt
    if importlib.metadata.version("iree-base-runtime") != "3.11.0":
        raise ValueError("This runtime requires IREE 3.11.0")
    if not device_uri.startswith("vulkan://") or not device_uri.removeprefix("vulkan://"):
        raise ValueError("An explicit Vulkan device URI is required")
    driver = rt.get_driver("vulkan")
    devices = driver.query_available_devices()
    selected = [d for d in devices if d["path"] == device_uri.removeprefix("vulkan://")]
    if len(selected) != 1:
        raise ValueError("The requested device URI must identify one enumerated Vulkan device")
    device = rt.get_device(device_uri, cache=False)
    result = {"protocol": PROTOCOL, "event": "preflight", "ready": True,
              "runtime_version": "3.11.0", "device_uri": device_uri,
              "device_name": selected[0]["name"], "cpu_model_fallback": False,
              "scope": "device_initialization_only"}
    del device, driver
    gc.collect()
    return result


def serve(session, stream):
    emit(session.ready())
    while True:
        raw = stream.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_REQUEST_BYTES or not raw.endswith("\n"):
            raise ValueError("Oversized or incomplete worker request")
        request = strict_json(raw)
        if not isinstance(request, dict):
            raise ValueError("Worker requests must be JSON objects")
        request_id = request.get("id")
        if type(request_id) is not int or request_id < 0:
            raise ValueError("Invalid request ID")
        operation = request.get("op")
        try:
            fields = {"id", "op", "tokens"} if operation == "forward" else {"id", "op"}
            if set(request) != fields:
                raise RequestError("Unexpected or missing worker request fields")
            if operation == "close":
                emit({"id": request_id, "ok": True, "position": session.position})
                return 0
            if operation == "reset":
                session.reset()
                response = {"id": request_id, "ok": True, "position": 0}
            elif operation == "forward":
                logits = session.forward(request.get("tokens"))
                response = {"id": request_id, "ok": True, "position": session.position,
                            "completed_device_steps": session.steps,
                            "logits_f32le": base64.b64encode(logits.astype("<f4", copy=False).tobytes()).decode("ascii")}
            else:
                raise RequestError("Unsupported worker operation")
            emit(response)
        except RequestError as error:
            # Input checks happen before any model state mutation.
            emit({"id": request_id, "ok": False, "position": session.position, "error": str(error), "fatal": False})
        except Exception as error:
            # A failed dispatch may have partially advanced a batch. Never reuse that state.
            emit({"id": request_id, "ok": False, "position": session.position, "error": str(error), "fatal": True})
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--probe", action="store_true", help="Check the selected device and runtime without loading a model")
    parser.add_argument("--python-package-directory", type=Path)
    parser.add_argument("--allow-cpu-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.python_package_directory:
        sys.path.insert(0, str(args.python_package_directory.resolve(strict=True)))
    if args.probe:
        if args.allow_cpu_for_tests or args.bundle:
            parser.error("--probe requires only the device and optional private packages")
        emit(probe(args.device))
        return 0
    if args.bundle is None:
        parser.error("--bundle is required for model execution")
    session = IreeSession(args.bundle, args.device, allow_cpu_for_tests=args.allow_cpu_for_tests)
    try:
        return serve(session, sys.stdin)
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
