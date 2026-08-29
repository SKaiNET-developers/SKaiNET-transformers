import java.util.zip.ZipFile

plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
    application
}

application {
    mainClass.set("sk.ainet.apps.skainet.cli.MainKt")
}

dependencies {
    implementation(platform(project(":llm-bom")))

    // Core
    implementation(project(":llm-core"))
    implementation(project(":llm-agent"))

    // Model runtimes (all architectures)
    implementation(project(":llm-runtime:kllama"))

    // Inference modules (for network loaders)
    implementation(project(":llm-inference:llama"))
    implementation(project(":llm-inference:qwen"))
    implementation(project(":llm-inference:gemma"))
    implementation(project(":llm-inference:apertus"))

    // SKaiNET core libraries
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.backend.api)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.backend.nativeCpu)
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

    // Workaround for com.gradleup.shadow:9.4.x — mergeServiceFiles()
    // silently drops one of two co-located META-INF/services/<X>
    // entries when both skainet-backend-cpu and skainet-backend-
    // native-cpu are on the classpath. skainet-cli pulls the kllama
    // runtime, which (in 0.22.0+) brings native-cpu in transitively;
    // without this fix the resulting shadow JAR runs Panama priority-
    // 50 even when the native lib is bundled. See PR #88 for the
    // kllama-cli copy and the underlying repro. Drop when the
    // upstream shadow merge is fixed.
    val skainetCliRuntimeClasspathFiles = project.configurations.named("runtimeClasspath")
        .map { it.files.filter { f -> f.name.endsWith(".jar") } }
    doLast {
        val jar = archiveFile.get().asFile
        val servicePath = "META-INF/services/sk.ainet.backend.api.kernel.KernelProvider"
        val entries = linkedSetOf<String>()
        for (cpJar in skainetCliRuntimeClasspathFiles.get()) {
            ZipFile(cpJar).use { zf ->
                val zipEntry = zf.getEntry(servicePath)
                if (zipEntry != null) {
                    zf.getInputStream(zipEntry).bufferedReader().useLines { lines ->
                        lines.map { it.trim() }
                            .filter { it.isNotEmpty() && !it.startsWith("#") }
                            .forEach { entries.add(it) }
                    }
                }
            }
        }
        if (entries.isEmpty()) return@doLast
        val tmpFile = temporaryDir.resolve("kernel-provider-services.txt")
        tmpFile.writeText(entries.joinToString("\n", postfix = "\n"))
        ant.withGroovyBuilder {
            "zip"(
                "destfile" to jar.absolutePath,
                "update" to true,
            ) {
                "zipfileset"(
                    "file" to tmpFile.absolutePath,
                    "fullpath" to servicePath,
                )
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx42g", "-XX:MaxDirectMemorySize=42g")
}
