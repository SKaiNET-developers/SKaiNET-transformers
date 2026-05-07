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
    testImplementation(project(":llm-inference:bert"))

    // Use the BOM to pin SKaiNET versions. Coords intentionally version-less so
    // a broken BOM fails the build here instead of going unnoticed.
    testImplementation(platform(project(":llm-bom")))
    testImplementation("sk.ainet.core:skainet-lang-core")
    testImplementation("sk.ainet.core:skainet-backend-cpu")
    testImplementation("sk.ainet.core:skainet-io-gguf")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs = listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
    )
    minHeapSize = "2g"
    maxHeapSize = "16g"
}
