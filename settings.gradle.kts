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

// Temporary composite build for validating local SKaiNET fixes
// (executor liveness freeing + ofAuto leak fix). Remove once shipped.
val localSkaiNet = file("../SKaiNET")
if (localSkaiNet.isDirectory) {
    includeBuild(localSkaiNet)
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
include("llm-runtime:kqwen")
include("llm-runtime:kapertus")
include("llm-performance")
include("llm-apps:skainet-cli")
include("llm-apps:kllama-cli")
include("llm-apps:kbert-cli")
include("llm-apps:kapertus-cli")
include("llm-apps:kvoxtral-cli")
include("llm-bom")
