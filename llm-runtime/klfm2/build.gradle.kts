import java.io.File
import java.security.MessageDigest
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.*

plugins {
    id("distribution")
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

kotlin {
    jvm()
    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
        }
        jvmMain.dependencies {
            api(project(":llm-inference:lfm2"))
            api(project(":llm-core"))
            implementation(libs.skainet.frontend.text)
            implementation(libs.kotlinx.serialization.json)
        }
        jvmTest.dependencies {
            implementation(kotlin("test-junit5"))
            implementation(libs.junit.jupiter)
            runtimeOnly(libs.junit.platform.launcher)
            implementation(libs.skainet.frontend.text)
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = if (providers.gradleProperty("includeIntegration").isPresent) "8g" else "512m"
}

/** Provenance travels with the published JAR, including runtime source additions. */
abstract class Klfm2SourcePins : DefaultTask() {
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sources: ConfigurableFileCollection
    @get:Internal abstract val repositoryDirectory: Property<String>
    @get:OutputFile abstract val outputFile: RegularFileProperty

    @TaskAction fun writePins() {
        val root = File(repositoryDirectory.get()).toPath()
        fun quote(value: String) = "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        val records = sources.files.sortedBy { root.relativize(it.toPath()).toString() }.map { source ->
            val path = root.relativize(source.toPath()).toString().replace(File.separatorChar, '/')
            val hash = MessageDigest.getInstance("SHA-256").digest(source.readBytes()).joinToString("") { "%02x".format(it) }
            "{\"file\":" + quote(path) + ",\"sha256\":" + quote(hash) + "}"
        }
        val output = outputFile.get().asFile
        output.parentFile.mkdirs()
        output.writeText("{\"schema_version\":1,\"engine_version\":\"0.62.0\",\"files\":[" + records.joinToString(",") + "]}\n")
    }
}

val generateSourcePins = tasks.register<Klfm2SourcePins>("generateKlfm2SourcePins") {
    repositoryDirectory.set(rootProject.layout.projectDirectory.asFile.absolutePath)
    sources.from(fileTree("src/jvmMain") { include("**/*.kt", "**/*.py", "scripts/*") },
        layout.projectDirectory.file("build.gradle.kts"), layout.projectDirectory.file("gradle.properties"),
        rootProject.layout.projectDirectory.file("build.gradle.kts"), rootProject.layout.projectDirectory.file("gradle.properties"),
        rootProject.layout.projectDirectory.file("gradle/libs.versions.toml"),
        rootProject.layout.projectDirectory.file("llm-bom/build.gradle.kts"))
    outputFile.set(layout.buildDirectory.file("generated/source-pins/klfm2-source-pins.json"))
}
kotlin.sourceSets.named("jvmMain") { resources.srcDir(generateSourcePins.map { it.outputFile.get().asFile.parentFile }) }


val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")
distributions {
    main {
        distributionBaseName.set("klfm2-iree-chat")
        contents {
            into("lib") {
                from(tasks.named("jvmJar"))
                from(jvmMainCompilation.runtimeDependencyFiles)
            }
            into("bin") {
                from("src/jvmMain/scripts")
                filePermissions { unix("rwxr-xr-x") }
            }
            from("README.md")
            from(rootProject.layout.projectDirectory.file("LICENCE"))
        }
    }
}
tasks.register<JavaExec>("runIreeChat") {
    group = "application"
    description = "Run the installed optional IREE chat distribution (use --args='--help')"
    dependsOn(tasks.named("installDist"))
    mainClass.set("sk.ainet.apps.klfm2.Lfm2IreeChatCli")
    classpath = fileTree(layout.buildDirectory.dir("install/klfm2-iree-chat/lib")) { include("*.jar") }
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "512m"
}

// Running Python tests must never add host-specific bytecode to published resources.
tasks.named<Copy>("jvmProcessResources") {
    exclude("**/__pycache__/**", "**/*.pyc")
}
