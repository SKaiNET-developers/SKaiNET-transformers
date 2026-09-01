pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        // Local-dev lane: `-PskainetMavenLocal` resolves sk.ainet artifacts from
        // ~/.m2 (an unsigned `publishToMavenLocal` of the SKaiNET repo) ahead of
        // Maven Central — how unreleased engine features are validated here.
        if (providers.gradleProperty("skainetMavenLocal").isPresent) {
            mavenLocal { content { includeGroupByRegex("""sk\.ainet.*""") } }
        }
        google()
        mavenCentral()
    }
}

// Composite build for validating local SKaiNET fixes against this repo.
// Opt-in: run with -PuseLocalSkainet=true (or set useLocalSkainet=true in
// gradle.properties) to substitute sk.ainet.core:* with a sibling ../SKaiNET
// checkout instead of the published Maven artifacts. Off by default.
if (providers.gradleProperty("useLocalSkainet").orNull == "true") {
    includeBuild("../SKaiNET")
}

rootProject.name = "SKaiNET-transformers"

include("llm-api")
include("transformer-core")
include("llm-core")
include("llm-agent")
include("llm-providers")
include("llm-inference:llama")
include("llm-inference:qwen")
include("llm-inference:bitnet")
include("llm-inference:gemma")
// FunctionGemma function-calling product module: export spec + harness + contract
// manifest for the DSL -> StableHLO -> IREE pipeline (whisper/moonshine pattern).
// Depends on :llm-inference:gemma for the architecture; owns the export contract.
include("llm-inference:functiongemma")
// SmolLM2 compiled-export product module (transformers#305): the redecode
// (fixed-seq, in-graph argMax) StableHLO export for the compiled leg of the
// SmolLM2 cross-target reproducer. Depends on :llm-inference:llama for the
// architecture; owns the export contract, mirroring functiongemma's shape.
include("llm-inference:smollm2")
include("llm-inference:moonshine")
include("llm-inference:whisper")
include("llm-inference:apertus")
include("llm-inference:bert")
include("llm-inference:voxtral")
include("llm-inference:t5")
include("llm-inference:vec2text")
include("llm-runtime:kllama")
include("llm-runtime:kgemma")
// :llm-runtime:kqwen — removed; Qwen runs through the DSL Qwen path
// (`QwenNetworkLoader` + `OptimizedLLMRuntime`) since #121 (kllama CLI
// swap). The legacy `QwenIngestion` facade had no remaining consumers
// after the architectural refactor — see PR closing this module.
include("llm-runtime:kapertus")
include("llm-runtime:kbitnet")
// Gemma-on-IREE runtime: decode loop + iree-run-module driver + tool-call codec
// (the on-device side of the DSL -> StableHLO -> IREE path).
include("llm-runtime:gemma-iree")
// Generic Android JNI runtime for the DSL -> StableHLO -> IREE compiled path
// (transformers#305): drives ANY fixed-seq redecode vmfb with an in-graph
// argMax tail over the real IREE C API — model-agnostic, unlike gemma-iree
// (board-specific, subprocess-driven). :llm-inference:smollm2 is its first
// producer of a (vmfb, irpa, function-name) triple.
include("llm-runtime:iree-android")
include("llm-performance")
include("llm-apps:skainet-cli")
include("llm-apps:skainet-decode")
include("llm-apps:kllama-cli")
include("llm-apps:kllama-java-sample")
include("llm-apps:kbert-cli")
include("llm-test:llm-test-java")
include("llm-bom")
