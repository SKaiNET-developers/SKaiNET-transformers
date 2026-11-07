import java.security.MessageDigest
import java.io.File
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.*

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
}

// The qualified execution/loader surface is JVM CPU. Portable tensor blocks
// live in commonMain; extra targets require their own build/runtime validation.
kotlin {
    jvm()
    sourceSets {
        commonMain.dependencies {
            // api, not implementation: the api dependencies below take their versions from it in the POM.
            api(project.dependencies.platform(project(":llm-bom")))
            api(libs.skainet.lang.core)
            api(libs.skainet.backend.cpu)
            api(libs.kotlinx.serialization.json)
        }
        jvmMain.dependencies {
            api(project(":llm-core"))
            api(libs.skainet.compile.hlo)
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.compile.dag)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.kotlinx.coroutines)
        }
        jvmTest.dependencies {
            implementation(kotlin("test-junit5"))
            implementation(libs.junit.jupiter)
            runtimeOnly(libs.junit.platform.launcher)
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "2g"
    // The gated real-checkpoint decode-step test (LFM2_MODEL_DIR) loads LFM2.5-350M twice.
    if (System.getenv("LFM2_MODEL_DIR") != null) maxHeapSize = "12g"
    System.getenv("LFM2_MODEL_DIR")?.let { environment("LFM2_MODEL_DIR", it) }
}

/** Provenance travels with the published JAR, including runtime source additions. */
abstract class Lfm2SourcePins : DefaultTask() {
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

val generateSourcePins = tasks.register<Lfm2SourcePins>("generateLfm2SourcePins") {
    repositoryDirectory.set(rootProject.layout.projectDirectory.asFile.absolutePath)
    sources.from(fileTree("src/commonMain") { include("**/*.kt") }, fileTree("src/jvmMain") { include("**/*.kt") },
        layout.projectDirectory.file("build.gradle.kts"), layout.projectDirectory.file("gradle.properties"),
        rootProject.layout.projectDirectory.file("build.gradle.kts"), rootProject.layout.projectDirectory.file("gradle.properties"),
        rootProject.layout.projectDirectory.file("gradle/libs.versions.toml"),
        rootProject.layout.projectDirectory.file("llm-bom/build.gradle.kts"))
    outputFile.set(layout.buildDirectory.file("generated/source-pins/lfm2-source-pins.json"))
}
kotlin.sourceSets.named("jvmMain") { resources.srcDir(generateSourcePins.map { it.outputFile.get().asFile.parentFile }) }

val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")
tasks.register<JavaExec>("exportLfm2") {
    group = "export"
    description = "Export a local verified LFM2 checkpoint; use --args='--help' for options"
    mainClass.set("sk.ainet.models.lfm2.export.Lfm2ExportHarnessKt")
    classpath = files(jvmMainCompilation.output.allOutputs, jvmMainCompilation.runtimeDependencyFiles)
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
}

tasks.register<JavaExec>("exportLfm2DecodeStep") {
    group = "export"
    description = "Export the LFM2 decode step with block-quantized weights for an NPU; use --args='--help' for options"
    mainClass.set("sk.ainet.models.lfm2.export.Lfm2DecodeStepExportCli")
    classpath = files(jvmMainCompilation.output.allOutputs, jvmMainCompilation.runtimeDependencyFiles)
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
}

tasks.register<JavaExec>("exportLfm2Generation") {
    group = "export"
    description = "Export the fixed-capacity LFM2 cached generation graph"
    mainClass.set("sk.ainet.models.lfm2.export.Lfm2GenerationExportCli")
    classpath = files(jvmMainCompilation.output.allOutputs, jvmMainCompilation.runtimeDependencyFiles)
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
}
