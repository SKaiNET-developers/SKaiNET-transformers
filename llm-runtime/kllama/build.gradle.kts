import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    android {
        namespace = "sk.ainet.apps.kllama"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
        // On-device e2e inference spike (#272). Runs the SmolLM2 generation
        // loop on a connected device/emulator with the NEON JNI backend that
        // androidMain ships since 0.39.0; skips unless a model server is
        // reachable (see AndroidSmolLm2E2eTest).
        withDeviceTest {
            instrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        }
    }

    linuxX64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    linuxArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    // Library-only Apple targets: no CLI executable, consumers link the klib
    // into their app. src/iosMain (the registerPlatformBackends actual) existed
    // before these targets did and was silently dead code — this module sets
    // kotlin.mpp.applyDefaultHierarchyTemplate=false, so the source set must be
    // wired by hand below (#271).
    iosArm64()
    iosSimulatorArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(project(":llm-inference:llama"))
            implementation(project(":llm-inference:qwen"))
            implementation(project(":llm-agent"))
            // api, not implementation: GenerationConfig.prefillStrategy exposes
            // llm-core's PrefillStrategy in a public signature (see #226).
            api(project(":llm-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.lang.ksp.annotations)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.serialization.json)

        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
            // runTest for the cross-target inference spike — runBlocking does
            // not exist on the JS/Wasm targets.
            implementation(libs.kotlinx.coroutines.test)
            implementation(libs.skainet.lang.models)
            implementation(libs.skainet.io.gguf)
        }

        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                // :llm-inference:qwen now in production deps for the CLI
                // swap; no separate test-scope entry needed.
            }
        }
        // val androidMain by getting
        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting {
                dependencies {
                    // NEON kernel backend (AAR, engine >= 0.39.0). ServiceLoader
                    // self-registers on ART; runtime classpath is all it needs.
                    runtimeOnly(libs.skainet.backend.jniCpu)
                }
            }
        }
        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }
        if (!project.hasProperty("buildFatJar")) {
            val androidDeviceTest by getting {
                dependencies {
                    implementation(libs.androidx.test.junit)
                    implementation(libs.androidx.test.runner)
                    implementation(libs.kotlinx.coroutines)
                }
            }
        }

        val nativeMain by creating {
            dependsOn(commonMain.get())
        }
        val linuxMain by creating { dependsOn(nativeMain) }
        val macosMain by creating { dependsOn(nativeMain) }
        val iosMain by creating { dependsOn(nativeMain) }
        val linuxX64Main by getting { dependsOn(linuxMain) }
        val linuxArm64Main by getting { dependsOn(linuxMain) }
        val macosArm64Main by getting { dependsOn(macosMain) }
        val iosArm64Main by getting { dependsOn(iosMain) }
        val iosSimulatorArm64Main by getting { dependsOn(iosMain) }

        // Native test tree — mirrors nativeMain, so the cross-target inference
        // spike (SmolLm2InferenceSpike, commonTest) and its posix `readEnv`
        // actual compile/run on the K/N targets. linuxX64Test runs natively on
        // a Linux CI host; iosSimulatorArm64Test runs on a Mac. Manual wiring
        // because this module sets applyDefaultHierarchyTemplate=false (#271).
        val nativeTest by creating { dependsOn(commonTest.get()) }
        val linuxX64Test by getting { dependsOn(nativeTest) }
        val linuxArm64Test by getting { dependsOn(nativeTest) }
        val macosArm64Test by getting { dependsOn(nativeTest) }
        val iosArm64Test by getting { dependsOn(nativeTest) }
        val iosSimulatorArm64Test by getting { dependsOn(nativeTest) }
    }
}

// Environment variables do not reach an iOS simulator test process unless
// prefixed with SIMCTL_CHILD_ — simctl strips that prefix when it spawns the
// process. Forward the spike's model-path var so iosSimulatorArm64Test can be
// gated on it (transformers#272).
tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
    listOf("SMOLLM2_MODEL", "SMOLLM2_QUANT", "SMOLLM2_STEPS").forEach { key ->
        providers.environmentVariable(key).orNull?.let { environment("SIMCTL_CHILD_$key", it) }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
    maxHeapSize = "32g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.register<JavaExec>("runJvm") {
    description = "Run kllama CLI on JVM"
    group = "application"
    mainClass.set("sk.ainet.apps.kllama.cli.MainKt")
    classpath = files(
        kotlin.jvm().compilations["main"].output.allOutputs,
        configurations["jvmRuntimeClasspath"]
    )
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
