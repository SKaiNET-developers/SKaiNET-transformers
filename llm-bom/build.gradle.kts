plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "sk.ainet.llm"
version = rootProject.findProperty("VERSION_NAME") ?: "0.1.0"

javaPlatform {
    allowDependencies()
}

dependencies {
    // Import SKaiNET BOM so consumers get aligned SKaiNET versions too
    api(platform("sk.ainet:skainet-bom:${libs.versions.skainet.get()}"))

    constraints {
        // LLM core
        api(project(":llm-core"))

        // Agent
        api(project(":llm-agent"))

        // Inference — model loaders
        api(project(":llm-inference:llama"))
        api(project(":llm-inference:qwen"))
        api(project(":llm-inference:gemma"))
        api(project(":llm-inference:bert"))

        // Runtime
        api(project(":llm-runtime:kllama"))
        api(project(":llm-runtime:kgemma"))
    }
}
