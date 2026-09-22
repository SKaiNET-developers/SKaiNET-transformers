#!/usr/bin/env bash
# Builds libskainet_moonshine_stream.so — the streaming Moonshine v2 ASR runtime (five DSL graphs on the
# IREE runtime, Vulkan or local-task) — for one Android ABI with the pinned skainet/iree-android:3.11.0
# image, the same way build-iree-kv.sh builds the KV session runtime.
#
#   native/build-moonshine-stream.sh arm64-v8a   [--vulkan] [cache-dir]
#   native/build-moonshine-stream.sh armeabi-v7a [--vulkan] [cache-dir]
#
# The IREE host tools and the per-ABI runtime tree are cached under
# ~/.cache/skainet-iree/3.11.0 (override with IREE_CACHE or the third argument); a warm cache
# builds in seconds. Output: native/out/libskainet_moonshine_stream.so — copy it to
# src/main/jniLibs/<abi>/ to ship it.
#
# Geometry is compiled in (tiny: dim 320 / 6 layers / 8 heads / head 40, the default). The `small`
# checkpoints (encoder 620 / decoder 512 / 10 layers / head 64) need a separately built variant:
#   GEOMETRY="-DL=10 -DHD=64 -DDIM=512 -DENC_DIM=620" native/build-moonshine-stream.sh <abi> --vulkan
set -euo pipefail
ABI="${1:-arm64-v8a}"; shift || true
VULKAN=""; EXTRA=()
[ -n "${GEOMETRY:-}" ] && EXTRA=(--cmake "-DCMAKE_C_FLAGS=${GEOMETRY}")
while [ $# -gt 0 ]; do
  case "$1" in
    --vulkan) VULKAN="--vulkan" ;;
    *) CACHE="$1" ;;
  esac
  shift
done
CACHE="${CACHE:-${IREE_CACHE:-$HOME/.cache/skainet-iree/3.11.0}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/out" "$CACHE/build-host" "$CACHE/build-$ABI"
docker run --rm \
  -v "$HERE":/src:ro -v "$HERE/out":/out \
  -v "$CACHE/build-host":/iree/build-host \
  -v "$CACHE/build-$ABI":/iree/build-$ABI \
  skainet/iree-android:3.11.0 \
  build-so "$ABI" --name skainet_moonshine_stream --src /src/moonshine_stream_jni.c $VULKAN ${EXTRA[@]+"${EXTRA[@]}"} \
  --link iree_modules_io_parameters_parameters \
  --link iree_io_parameter_index \
  --link iree_io_parameter_index_provider \
  --link iree_io_formats_irpa_irpa
