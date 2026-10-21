#!/usr/bin/env bash
# Builds libskainet_whisper_session.so — Whisper speech-to-text over the three graphs of WhisperExportCli
# (encoder, masked prefill, KV step) on the IREE runtime, Vulkan or local-task — for one Android ABI with the
# pinned skainet/iree-android:3.11.0 image, the same way build-moonshine-stream.sh builds the Moonshine runtime.
#
#   native/build-whisper-session.sh arm64-v8a   [--vulkan] [cache-dir]
#   native/build-whisper-session.sh armeabi-v7a [--vulkan] [cache-dir]
#
# Nothing about the checkpoint is compiled in (the geometry comes from the contract at runtime), so one build
# serves tiny, base, small and the English-only variants. Output: native/out/libskainet_whisper_session.so —
# copy it to src/main/jniLibs/<abi>/ to ship it.
set -euo pipefail
ABI="${1:-arm64-v8a}"; shift || true
VULKAN=""
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
  build-so "$ABI" --name skainet_whisper_session --src /src/whisper_session_jni.c $VULKAN \
  --link iree_modules_io_parameters_parameters \
  --link iree_io_parameter_index \
  --link iree_io_parameter_index_provider \
  --link iree_io_formats_irpa_irpa
