plugins {
    alias(libs.plugins.skainet.multiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

// Targets: gradle.properties has skainet.targets=jvm,js,wasmJs,wasmWasi,apple,linux (the
// default, NOT androidNative) because this module wants androidNativeArm32 only, not the
// androidNativeArm64 the plugin's androidNative group would also add -- androidNativeArm32() is
// declared manually below instead, alongside this module's own already-manual source-set
// hierarchy (kotlin.mpp.applyDefaultHierarchyTemplate=false, also in gradle.properties).
skainet {
    namespace = "sk.ainet.apps.llm"
    // sk.ainet.multiplatform defaults explicitApi to true; this module has a handful of
    // declarations (DecoderGgufWeightLoader.kt's GGUF key-naming helpers) missing visibility
    // modifiers -- pre-existing, harmless, but never checked before since explicit API mode was
    // never on here. Opting out rather than editing unrelated source as a side effect of this
    // build-tooling migration; worth turning back on as its own small follow-up.
    explicitApi = false
}

kotlin {
    androidNativeArm32()

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
            implementation(libs.skainet.io.safetensors)
            // KernelRegistry/KernelProvider capability queries for the
            // packed-quant kernel gate (`hasPackedMatmulKernel`, #170).
            // implementation-scoped: no backend types leak into the API.
            implementation(libs.skainet.backend.api)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.serialization.json)
        }

        val jvmMain by getting

        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlinx.coroutines)
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
        val androidNativeArm32Main by getting { dependsOn(nativeMain) }

        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting { dependsOn(registryBasedMain) }
        }
        val jsMain by getting { dependsOn(registryBasedMain) }
        val wasmJsMain by getting { dependsOn(registryBasedMain) }
        val wasmWasiMain by getting { dependsOn(registryBasedMain) }
    }
}
