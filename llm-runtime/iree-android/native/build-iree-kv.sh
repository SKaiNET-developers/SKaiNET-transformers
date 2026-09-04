#!/usr/bin/env bash
# Cross-build libskainet_iree_kv.so for an Android ABI using the consolidated
# skainet/iree-android image from SKaiNET-iree-toolchain (IREE v3.11 runtime tree + NDK
# r27c; build it there with `make android`). The build-so subcommand injects the CMake
# target, configures the local-task/local-sync CPU drivers (+ Vulkan when --vulkan is
# passed), builds and strips.
#
# This is a GENERIC IREE redecode runtime (llm-runtime/iree-android) — it knows nothing
# about any specific model. Weights for whatever vmfb it's pointed at are EXTERNAL (an
# `.irpa` loaded at session-create time via the io_parameters VM module), so the `.so`
# needs the io_parameters VM module and its iree/io dependencies explicitly linked — the
# default `iree_runtime_unified` target alone does not pull them in. This requirement is a
# property of the runtime (any external-weights vmfb needs it), not of any one model.
#
#   ./build-iree-redecode.sh [armeabi-v7a|arm64-v8a] [--vulkan] [cache-dir]
#
# Cache dir (default $IREE_CACHE, or ~/.cache/skainet-iree/<ver>) warms /iree/build-host
# and /iree/build-<ABI> across runs and ABIs.
# Output: out/libskainet_iree_kv.so -> copy to
#   ../src/main/jniLibs/<ABI>/libskainet_iree_kv.so
set -euo pipefail
ABI="${1:-arm64-v8a}"
VULKAN=""
if [ "${2:-}" = "--vulkan" ]; then VULKAN="--vulkan"; shift; fi
CACHE="${2:-${IREE_CACHE:-$HOME/.cache/skainet-iree/3.11.0}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/out" "$CACHE/build-host" "$CACHE/build-$ABI"

docker run --rm \
  -v "$HERE":/src:ro -v "$HERE/out":/out \
  -v "$CACHE/build-host":/iree/build-host \
  -v "$CACHE/build-$ABI":/iree/build-$ABI \
  skainet/iree-android:3.11.0 \
  build-so "$ABI" --name skainet_iree_kv --src /src/iree_kv_jni.c $VULKAN \
  --link iree_modules_io_parameters_parameters \
  --link iree_io_parameter_index \
  --link iree_io_parameter_index_provider \
  --link iree_io_formats_irpa_irpa
