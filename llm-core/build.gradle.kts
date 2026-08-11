import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

kotlin {
    android {
        namespace = "sk.ainet.apps.llm"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    linuxX64()
    linuxArm64()
    macosArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            // BOM-only: every skainet-* alias is versionless; this platform
            // constraint (re-exporting sk.ainet:skainet-bom) supplies the
            // versions. Bumping the engine is then a one-line change at the
            // top of `gradle/libs.versions.toml`.
            implementation(project.dependencies.platform(project(":llm-bom")))
            api(project(":transformer-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.compile.dag)
            implementation(libs.skainet.compile.opt)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            // KernelRegistry/KernelProvider capability queries for the
            // packed-quant kernel gate (`hasPackedMatmulKernel`, #170).
            // implementation-scoped: no backend types leak into the API.
            implementation(libs.skainet.backend.api)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.serialization.json)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        val jvmMain by getting

        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                implementation(libs.junit)
                implementation(libs.skainet.io.gguf)
                implementation(libs.skainet.io.core)
                // CPU backend so jvmTest can actually run forward passes
                // against a real ExecutionContext (otherwise tensor ops are
                // VoidOps stubs). Mirrors the wiring in `llm-inference/*`.
                implementation(libs.skainet.backend.cpu)
            }
        }

        // Shared source set for all non-JVM targets (manual BackendRegistry)
        val registryBasedMain by creating {
            dependsOn(commonMain.get())
        }

        val nativeMain by creating { dependsOn(registryBasedMain) }
        val iosArm64Main by getting { dependsOn(nativeMain) }
        val iosSimulatorArm64Main by getting { dependsOn(nativeMain) }
        val linuxX64Main by getting { dependsOn(nativeMain) }
        val linuxArm64Main by getting { dependsOn(nativeMain) }
        val macosArm64Main by getting { dependsOn(nativeMain) }

        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting { dependsOn(registryBasedMain) }
        }
        val jsMain by getting { dependsOn(registryBasedMain) }
        val wasmJsMain by getting { dependsOn(registryBasedMain) }
        val wasmWasiMain by getting { dependsOn(registryBasedMain) }
    }
}
