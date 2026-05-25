plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    application
}

kotlin {
    jvmToolchain(21)
}

application {
    mainClass.set("sk.ainet.transformers.samples.kllama.Main")
    applicationDefaultJvmArgs = listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
    )
}

// Java-only sample: pick up sources from src/main/java
sourceSets["main"].java.srcDir("src/main/java")

dependencies {
    implementation(platform(project(":llm-bom")))

    implementation(project(":llm-runtime:kllama"))
    implementation(project(":llm-agent"))
    implementation(project(":llm-inference:llama"))

    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.io.gguf)
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    minHeapSize = "2g"
    maxHeapSize = "8g"
}
