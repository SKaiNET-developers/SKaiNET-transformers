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

// Composite build for validating local SKaiNET fixes against this repo. Auto-enables
// when ../SKaiNET is a sibling checkout. Pass -PuseLocalSkaiNet=false to opt out and
// resolve `sk.ainet.core:*` from mavenLocal / mavenCentral instead — useful for
// testing a published-to-mavenLocal version end-to-end.
val localSkaiNetEnabled = (settings.providers.gradleProperty("useLocalSkaiNet").orNull ?: "true").toBoolean()
val localSkaiNet = file("../SKaiNET")
if (localSkaiNetEnabled && localSkaiNet.isDirectory) {
    includeBuild(localSkaiNet)
}

rootProject.name = "SKaiNET-transformers"

include("llm-api")
include("llm-core")
include("llm-agent")
include("llm-providers")
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
include("llm-apps:kllama-java-sample")
include("llm-apps:kbert-cli")
include("llm-test:llm-test-java")
include("llm-bom")
