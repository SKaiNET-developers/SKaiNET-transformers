plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
    application
}

application {
    mainClass.set("sk.ainet.apps.voxtral.cli.MainKt")
}

dependencies {
    implementation(project(":llm-inference:voxtral"))
    implementation(project(":llm-inference:llama"))
    implementation(project(":llm-core"))
    implementation(project(":llm-runtime:kllama"))
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.gguf)
    implementation(libs.skainet.io.safetensors)
    implementation(libs.skainet.compile.core)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.kotlinx.io.core)
    implementation(libs.kotlinx.coroutines)
}

tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("kvoxtral")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.voxtral.cli.MainKt",
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
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
