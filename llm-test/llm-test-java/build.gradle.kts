plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
}

kotlin {
    jvmToolchain(21)
}

// Pick up Java sources alongside Kotlin in src/test
sourceSets["test"].java.srcDir("src/test/java")

dependencies {
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)

    // The Java surface under test
    testImplementation(project(":llm-runtime:kllama"))
    testImplementation(project(":llm-agent"))
    testImplementation(project(":llm-inference:llama"))

    // SKaiNET runtime needed by KLlamaJava (JVM target)
    testImplementation(libs.skainet.lang.core)
    testImplementation(libs.skainet.backend.cpu)
    testImplementation(libs.skainet.io.gguf)
}

tasks.test {
    useJUnitPlatform()
    jvmArgs = listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
    )
    minHeapSize = "2g"
    maxHeapSize = "8g"
}
