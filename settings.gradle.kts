pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
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
include("llm-inference:gemma")
include("llm-inference:moonshine")
include("llm-inference:apertus")
include("llm-inference:bert")
include("llm-inference:voxtral")
include("llm-runtime:kllama")
include("llm-runtime:kgemma")
// :llm-runtime:kqwen — removed; Qwen runs through the DSL Qwen path
// (`QwenNetworkLoader` + `OptimizedLLMRuntime`) since #121 (kllama CLI
// swap). The legacy `QwenIngestion` facade had no remaining consumers
// after the architectural refactor — see PR closing this module.
include("llm-runtime:kapertus")
// Gemma-on-IREE runtime: decode loop + iree-run-module driver + tool-call codec
// (the on-device side of the DSL -> StableHLO -> IREE path).
include("llm-runtime:gemma-iree")
include("llm-performance")
include("llm-apps:skainet-cli")
include("llm-apps:kllama-cli")
include("llm-apps:kllama-java-sample")
include("llm-apps:kbert-cli")
include("llm-test:llm-test-java")
include("llm-bom")
