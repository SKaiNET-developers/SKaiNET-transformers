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

rootProject.name = "SKaiNET-LLM"

include("llm-core")
include("llm-agent")
include("llm-inference:llama")
include("llm-inference:qwen")
include("llm-inference:gemma")
include("llm-inference:bert")
include("llm-runtime:kllama")
include("llm-runtime:kgemma")
include("llm-apps:kllama-cli")
include("llm-apps:kbert-cli")
include("llm-bom")
