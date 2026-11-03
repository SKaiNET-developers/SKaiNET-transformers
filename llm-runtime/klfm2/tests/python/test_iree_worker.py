"""Offline state-boundary regressions; no checkpoint, compiler or GPU required."""
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/jvmMain/resources/sk/ainet/apps/klfm2"))
import iree_worker as worker


def contract():
    states = [{"name": "history_0", "shape": [4, 3], "dtype": "f32"},
              {"name": "key_1", "shape": [1, 3, 4], "dtype": "f32"},
              {"name": "value_1", "shape": [1, 3, 4], "dtype": "f32"}]
    return {"schema_version": 1, "protocol": worker.PROTOCOL, "backend": "vulkan-spirv", "dtype": "float32",
            "embedding_policy": "one_hot_sharded", "capacity": 3,
            "dimensions": {"vocabulary": 7, "width": 4, "head_dimension": 4, "kv_heads": 1,
                           "conv_kernel": 3, "context_limit": 8, "rope_theta": 10000.0,
                           "layer_types": ["conv", "full_attention"]},
            "function_name": "lfm2_cached_step", "parameter_scope": "model",
            "inputs": [{"name": name, "shape": shape, "dtype": "f32"} for name, shape in
                       [("token_one_hot", [1, 7]), ("cos", [1, 4]), ("sin", [1, 4]),
                        ("mask", [1, 4]), ("write_mask", [1, 3, 1])]] + states,
            "outputs": [{"name": "logits", "shape": [1, 7], "dtype": "f32"}] + states}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = contract()
        for kind in ("module", "parameters"):
            path = self.root / kind
            path.write_bytes(kind.encode())
            self.manifest[kind] = {"file": kind, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def load(self):
        (self.root / "runtime-manifest.json").write_text(json.dumps(self.manifest))
        return worker.load_contract(self.root)

    def test_valid_manifest_and_artifact_binding(self):
        self.assertEqual(self.load()[0], self.manifest)
        (self.root / "module").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.load()

    def test_reordered_state_or_wrong_dtype_is_rejected(self):
        for part in ("inputs", "outputs"):
            original = copy.deepcopy(self.manifest[part])
            self.manifest[part][-1], self.manifest[part][-2] = self.manifest[part][-2], self.manifest[part][-1]
            with self.assertRaisesRegex(ValueError, "port order"):
                self.load()
            self.manifest[part] = original
        self.manifest["outputs"][0]["dtype"] = "f16"
        with self.assertRaisesRegex(ValueError, "port order"):
            self.load()

    def test_external_artifact_path_is_rejected(self):
        self.manifest["module"]["file"] = "/etc/hosts"
        with self.assertRaisesRegex(ValueError, "relative"):
            self.load()

    def test_invalid_capacity_and_rotary_geometry_are_rejected(self):
        for key, value in (("capacity", True), ("capacity", 0), ("capacity", 9)):
            self.manifest[key] = value
            with self.assertRaises(ValueError):
                self.load()
        self.manifest["capacity"] = 3
        self.manifest["dimensions"]["head_dimension"] = 3
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for value in ('{"x":1,"x":2}', '{"outer":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                worker.strict_json(value)

    def test_partial_initialization_releases_owned_references(self):
        session = worker.IreeSession.__new__(worker.IreeSession)
        def fail(owner, *args, **kwargs):
            owner.device = object()
            owner.states = [object()]
            raise RuntimeError("initialization failed")
        with patch.object(worker.IreeSession, "_initialize", fail):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                session.__init__("unused", "vulkan://synthetic")
        self.assertIsNone(session.device)
        self.assertEqual(session.states, [])


class FakeBuffer:
    def __init__(self, values, *, memory_type=1, usage=8, events=None):
        self.values = values
        self.memory_flags = memory_type
        self.usage = usage
        self.events = events if events is not None else []
    def memory_type(self):
        return self.memory_flags
    def allowed_usage(self):
        return self.usage
    def byte_length(self):
        return self.values.nbytes
    def map(self):
        self.events.append("map")
        return self.values


class FakeView:
    element_type = "f32"
    def __init__(self, buffer):
        self.buffer = buffer
    def get_buffer(self):
        return self.buffer
    def map(self):
        return self.buffer.map()


class FakeArray:
    def __init__(self, values, can_map=False, *, buffer=None):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.can_map = can_map
        self.buffer = buffer or FakeBuffer(self.values)
    @property
    def _buffer_view(self):
        if not self.can_map:
            raise AssertionError("State must never be copied to the host")
        return FakeView(self.buffer)
    def to_host(self):
        raise AssertionError("The leaking to_host/asarray path must never be used")


def fake_runtime():
    return types.SimpleNamespace(
        DeviceArray=FakeArray,
        asdevicearray=lambda device, value, **kwargs: FakeArray(value),
        MemoryType=types.SimpleNamespace(HOST_VISIBLE=1, HOST_LOCAL=3, DEVICE_VISIBLE=4),
        BufferUsage=types.SimpleNamespace(MAPPING_SCOPED=8, TRANSFER_TARGET=16),
        HalElementType=types.SimpleNamespace(map_to_dtype=lambda kind: np.dtype(np.float32)),
    )


class HostCopyTests(unittest.TestCase):
    def test_direct_mapping_returns_independent_fp32_storage_without_caching_host_view(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        array = FakeArray(values, can_map=True)
        result = worker.copy_device_f32_to_host(array, fake_runtime(), np, device=object())
        self.assertEqual(result.shape, values.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(np.shares_memory(values, result))
        values[:] = -1
        np.testing.assert_array_equal(result, np.arange(12, dtype=np.float32).reshape(3, 4))
        self.assertFalse(hasattr(array, "_host_array"))
        self.assertEqual(array.buffer.events, ["map"])

    def test_nonmappable_source_copies_and_waits_before_mapping(self):
        events = []
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        source = FakeBuffer(values, memory_type=0, usage=0, events=events)
        array = FakeArray(values, can_map=True, buffer=source)
        rt = fake_runtime()
        class Fence:
            @staticmethod
            def create_at(semaphore, value):
                return Fence(semaphore, value)
            def __init__(self, semaphore, value):
                self.semaphore, self.value = semaphore, value
            def wait(self):
                events.append("wait")
                return self.semaphore.value >= self.value
        rt.HalFence = Fence
        allocation = {}
        def allocate_buffer(**kwargs):
            allocation.update(kwargs)
            events.append("allocate")
            return FakeBuffer(np.zeros(kwargs["allocation_size"] // 4, dtype=np.float32), events=events)
        def queue_copy(source_buffer, target_buffer, *, wait_semaphores, signal_semaphores):
            events.append("copy")
            self.assertEqual(wait_semaphores.value, 0)
            target_buffer.values[:] = source_buffer.values.reshape(-1)
            signal_semaphores.semaphore.value = signal_semaphores.value
        device = types.SimpleNamespace(allocator=types.SimpleNamespace(allocate_buffer=allocate_buffer),
                                       create_semaphore=lambda value: types.SimpleNamespace(value=value), queue_copy=queue_copy)
        result = worker.copy_device_f32_to_host(array, rt, np, device=device)
        np.testing.assert_array_equal(result, values)
        self.assertFalse(np.shares_memory(result, values))
        self.assertEqual(events, ["allocate", "copy", "wait", "map"])
        self.assertEqual(allocation, {"memory_type": 7, "allowed_usage": 24, "allocation_size": values.nbytes})

    def test_wrong_dtype_and_truncated_mapped_storage_are_rejected(self):
        rt = fake_runtime()
        array = FakeArray(np.arange(4, dtype=np.float32), can_map=True)
        array.dtype = np.float16
        with self.assertRaisesRegex(ValueError, "FP32"):
            worker.copy_device_f32_to_host(array, rt, np, device=object())
        array.dtype = np.float32
        array.shape = (8,)
        with self.assertRaisesRegex(ValueError, "smaller"):
            worker.copy_device_f32_to_host(array, rt, np, device=object())


class StateTests(unittest.TestCase):
    def setUp(self):
        self.session = worker.IreeSession.__new__(worker.IreeSession)
        s = self.session
        s.contract = contract()
        s.np = np
        s.rt = fake_runtime()
        s.device = object()
        s.capacity = 3
        s.vocabulary = 7
        s.dims = s.contract["dimensions"]
        s.state_ports = s.contract["outputs"][1:]
        s.steps = 0
        s.reset()
        self.calls = []
        def forward(*inputs):
            self.calls.append(inputs)
            return (FakeArray(inputs[0].values, can_map=True), *[FakeArray(x.values + 1) for x in inputs[5:]])
        s.function = forward

    def test_prompt_and_decode_reuse_device_state_without_mapping(self):
        s = self.session
        initial = s.states.copy()
        np.testing.assert_array_equal(s.forward([2, 4]), np.eye(7, dtype=np.float32)[4])
        self.assertEqual(s.position, 2)
        self.assertIs(self.calls[0][5], initial[0])
        self.assertIsNot(self.calls[1][5], initial[0])
        np.testing.assert_array_equal(self.calls[1][5].values, np.ones((4, 3)))
        previous = s.states.copy()
        s.forward([1])
        self.assertIs(self.calls[-1][5], previous[0])
        self.assertEqual(s.position, 3)

    def test_position_masks_and_split_half_rope(self):
        s = self.session
        controls = s.controls(2)
        np.testing.assert_array_equal(controls[1], np.ones((1, 4)))
        np.testing.assert_array_equal(controls[2], np.zeros((1, 4)))
        self.assertEqual(np.flatnonzero(controls[3] == 0).tolist(), [3])
        self.assertEqual(np.flatnonzero(controls[4]).tolist(), [0])
        s.position = 2
        controls = s.controls(6)
        self.assertEqual(np.flatnonzero(controls[3] == 0).tolist(), [0, 1, 3])
        self.assertEqual(np.flatnonzero(controls[4]).tolist(), [2])
        self.assertEqual(controls[1][0, 0], controls[1][0, 2])
        self.assertEqual(controls[2][0, 1], controls[2][0, 3])

    def test_invalid_batch_is_atomic_and_context_cannot_wrap(self):
        s = self.session
        for tokens in ([], [1, 7], [True], [1.0], [1, 2, 3, 4]):
            with self.assertRaises(worker.RequestError):
                s.forward(tokens)
            self.assertEqual(s.position, 0)
            self.assertEqual(len(self.calls), 0)
        s.forward([1, 2, 3])
        with self.assertRaises(worker.RequestError):
            s.forward([1])
        self.assertEqual(s.position, 3)

    def test_reset_replaces_all_state_and_restarts_position(self):
        s = self.session
        s.forward([2])
        previous = s.states.copy()
        s.reset()
        self.assertEqual(s.position, 0)
        for old, new in zip(previous, s.states):
            self.assertIsNot(old, new)
            self.assertFalse(new.values.any())

    def test_returned_logits_own_host_memory(self):
        s = self.session
        device_logits = np.arange(7, dtype=np.float32).reshape(1, 7)
        s.function = lambda *args: (FakeArray(device_logits, can_map=True), *s.states)
        returned = s.forward([1])
        self.assertFalse(np.shares_memory(returned, device_logits))
        device_logits[:] = -100
        np.testing.assert_array_equal(returned, np.arange(7, dtype=np.float32))

    def test_nonfinite_logits_do_not_commit_failed_step(self):
        s = self.session
        s.function = lambda *args: (FakeArray(np.full((1, 7), np.nan), can_map=True), *s.states)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            s.forward([1])
        self.assertEqual(s.position, 0)

    def test_dispatch_value_error_is_fatal_not_an_input_rejection(self):
        s = self.session
        s.ready = lambda: {"event": "ready"}
        def fail(*args):
            raise ValueError("dispatch error")
        s.function = fail
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(worker.serve(s, io.StringIO('{"id":1,"op":"forward","tokens":[1]}\n')), 1)
        self.assertTrue(json.loads(output.getvalue().splitlines()[-1])["fatal"])

    def test_protocol_rejects_bad_input_then_accepts_fresh_forward(self):
        s = self.session
        s.ready = lambda: {"event": "ready"}
        requests = [{"id": 1, "op": "forward", "tokens": [9]},
                    {"id": 2, "op": "forward", "tokens": [2]}, {"id": 3, "op": "close"}]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(worker.serve(s, io.StringIO("".join(json.dumps(x) + "\n" for x in requests))), 0)
        lines = [json.loads(x) for x in output.getvalue().splitlines()]
        self.assertFalse(lines[1]["fatal"])
        self.assertEqual(lines[2]["position"], 1)
        values = np.frombuffer(base64.b64decode(lines[2]["logits_f32le"]), dtype="<f4")
        self.assertEqual(values.argmax(), 2)

    def test_unknown_request_fields_are_rejected_before_forward(self):
        s = self.session
        s.ready = lambda: {"event": "ready"}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(worker.serve(s, io.StringIO('{"id":1,"op":"forward","tokens":[2],"ignored":true}\n')), 0)
        self.assertFalse(json.loads(output.getvalue().splitlines()[-1])["ok"])
        self.assertEqual(s.position, 0)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
