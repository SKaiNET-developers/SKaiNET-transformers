plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
}

dependencies {
    implementation(project(":llm-inference:bert"))
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.safetensors)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.kotlinx.coroutines)
}

tasks.named<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJar") {
    archiveBaseName.set("kbert")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.bert.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true"
        )
    }

    mergeServiceFiles()
}

tasks.register<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJarDemo") {
    archiveBaseName.set("kbert-demo")
    archiveClassifier.set("all")
    archiveVersion.set("")
    from(sourceSets.main.get().output)
    configurations = listOf(project.configurations.runtimeClasspath.get())

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.bert.cli.DemoKt",
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
