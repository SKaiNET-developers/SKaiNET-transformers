pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "SKaiNET-transformers"

include("llm-core")
include("llm-agent")
include("llm-inference:llama")
include("llm-inference:qwen")
include("llm-inference:gemma")
include("llm-inference:apertus")
include("llm-inference:bert")
include("llm-inference:voxtral")
include("llm-runtime:kllama")
include("llm-runtime:kgemma")
include("llm-runtime:kapertus")
include("llm-performance")
include("llm-apps:kllama-cli")
include("llm-apps:kbert-cli")
include("llm-apps:kapertus-cli")
include("llm-apps:kvoxtral-cli")
include("llm-bom")
