import java.util.zip.ZipFile

plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
    application
}

application {
    mainClass.set("sk.ainet.apps.kllama.cli.MainKt")
}

dependencies {
    implementation(project(":llm-runtime:kllama"))
}

tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("kllama")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.kllama.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true"
        )
    }

    // Merge service files for proper SPI support.
    mergeServiceFiles()

    // Workaround: shadow's mergeServiceFiles silently drops the
    // skainet-backend-native-cpu KernelProvider entry on this version
    // (com.gradleup.shadow:9.4.1) — only the cpu module's
    // Scalar+PanamaVector lines survive. Reconcile by re-reading every
    // dependency JAR's services file and rewriting the merged file
    // with the union of all entries (deduplicated, newline-terminated).
    val runtimeClasspathFiles = project.configurations.named("runtimeClasspath")
        .map { it.files.filter { f -> f.name.endsWith(".jar") } }
    doLast {
        val jar = archiveFile.get().asFile
        val servicePath = "META-INF/services/sk.ainet.backend.api.kernel.KernelProvider"
        val entries = linkedSetOf<String>()
        for (cpJar in runtimeClasspathFiles.get()) {
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
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx42g", "-XX:MaxDirectMemorySize=64g")
}
