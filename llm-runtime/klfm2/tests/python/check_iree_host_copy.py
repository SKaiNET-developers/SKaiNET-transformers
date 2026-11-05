"""Optional IREE 3.11.0 public-HAL ownership check; CPU toy buffers only.

Run in a process with pinned IREE and NumPy installed. Capture stderr as well as
stdout: success requires no nanobind leaked-instance diagnostics at shutdown.
No checkpoint, compiler or GPU is required.
"""
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import types

import numpy as np
import iree.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/jvmMain/resources/sk/ainet/apps/klfm2"))
from iree_worker import copy_device_f32_to_host


def check():
    assert importlib.metadata.version("iree-base-runtime") == "3.11.0"
    device = rt.get_device("local-task", cache=False)
    copies = []
    class DeviceCopyObserver:
        allocator = device.allocator
        def create_semaphore(self, value):
            return device.create_semaphore(value)
        def queue_copy(self, source, target, **kwargs):
            copies.append(source.byte_length())
            return device.queue_copy(source, target, **kwargs)
    observed = DeviceCopyObserver()
    # local-task promotes allocations to mappable host memory. Override only
    # the capability predicate so the staging route still executes against
    # real HAL buffers, allocator, queue_copy, semaphores, fences and mappings.
    staging_rt = types.SimpleNamespace(DeviceArray=rt.DeviceArray, HalElementType=rt.HalElementType,
        BufferUsage=rt.BufferUsage, HalFence=rt.HalFence,
        MemoryType=types.SimpleNamespace(HOST_VISIBLE=1 << 63,
            HOST_LOCAL=rt.MemoryType.HOST_LOCAL, DEVICE_VISIBLE=rt.MemoryType.DEVICE_VISIBLE))
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    direct_reads = 0
    for index in range(200):
        values = expected + index
        direct = rt.asdevicearray(device, values, implicit_host_transfer=False)
        direct_buffer = direct._buffer_view.get_buffer()
        assert direct_buffer.memory_type() & int(rt.MemoryType.HOST_VISIBLE)
        assert direct_buffer.allowed_usage() & int(rt.BufferUsage.MAPPING_SCOPED)
        host = copy_device_f32_to_host(direct, rt, np, device=observed)
        np.testing.assert_array_equal(host, values)
        assert not np.shares_memory(host, values)
        assert not direct.is_host_accessible  # helper never stores a cached host view
        direct_reads += 1
        view = device.allocator.allocate_buffer_copy(
            memory_type=rt.MemoryType.DEVICE_LOCAL,
            allowed_usage=rt.BufferUsage.TRANSFER_SOURCE,
            device=device, buffer=values, element_type=rt.HalElementType.FLOAT_32,
        )
        nonmappable = rt.DeviceArray(device, view, implicit_host_transfer=False)
        copied = copy_device_f32_to_host(nonmappable, staging_rt, np, device=observed)
        np.testing.assert_array_equal(copied, values)
        assert not np.shares_memory(copied, values)
        assert not nonmappable.is_host_accessible
        host[:] = -1
        copied[:] = -2
        np.testing.assert_array_equal(values, expected + index)
    assert len(copies) == 200 and all(size == expected.nbytes for size in copies)
    return {"status": "passed_public_hal_cpu_only", "runtime_version": importlib.metadata.version("iree-base-runtime"),
            "numpy_version": np.__version__, "direct_maps": direct_reads, "queued_copies": len(copies),
            "owned_host_results": direct_reads + len(copies),
            "staging_copy_capability_predicate": "forced_for_cpu_test_only_real_public_hal_copy"}


if __name__ == "__main__":
    result = check()
    gc.collect()
    print(json.dumps(result))
