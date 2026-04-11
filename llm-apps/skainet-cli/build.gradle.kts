plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
    application
}

application {
    mainClass.set("sk.ainet.apps.skainet.cli.MainKt")
}

dependencies {
    // Core
    implementation(project(":llm-core"))
    implementation(project(":llm-agent"))

    // Model runtimes (all architectures)
    implementation(project(":llm-runtime:kllama"))

    // Inference modules (for network loaders)
    implementation(project(":llm-inference:llama"))

    // SKaiNET core libraries
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.gguf)
    implementation(libs.kotlinx.io.core)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.kotlinx.serialization.json)
}

tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("skainet")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.skainet.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true"
        )
    }

    mergeServiceFiles()
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx12g", "-XX:MaxDirectMemorySize=64g")
}
