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
            api(kotlin("stdlib-common"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.compile.dag)
            implementation(libs.skainet.compile.opt)
            implementation(libs.skainet.io.core)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        val jvmMain by getting

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
