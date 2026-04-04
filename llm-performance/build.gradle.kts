import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.shadow)
}

kotlin {
    android {
        namespace = "sk.ainet.performance"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm {
        mainRun {
            mainClass.set("sk.ainet.performance.cli.MainKt")
        }
    }

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.performance.cli.main"
                baseName = "llm-performance"

                val neuroBase = rootProject.projectDir.resolve("../../neuroSKai/neuroSK/projects").canonicalPath

                // Metal bridge
                val metalBridgeDir = providers.environmentVariable("SKAINET_METAL_BRIDGE_DIR")
                    .orElse(providers.gradleProperty("skainet.metal.bridge.dir"))
                    .orElse("$neuroBase/metal/neuroSKai-backend-metal/skainet-backends/skainet-backend-metal/build/native/bridge")
                    .get()

                // MLX bridge
                val mlxBridgeDir = providers.environmentVariable("SKAINET_MLX_BRIDGE_DIR")
                    .orElse(providers.gradleProperty("skainet.mlx.bridge.dir"))
                    .orElse("$neuroBase/mlx/neuroSKai-backend-mlx/skainet-backends/skainet-backend-mlx/build/native/bridge")
                    .get()

                // MLX system library
                val mlxLibDir = providers.environmentVariable("MLX_ROOT")
                    .map { "$it/lib" }
                    .orElse("/opt/homebrew/opt/mlx/lib")
                    .get()

                linkerOpts(
                    // Metal bridge
                    "-L$metalBridgeDir", "-lmetal_bridge",
                    // MLX bridge
                    "-L$mlxBridgeDir", "-lmlx_bridge",
                    // MLX system library
                    "-L$mlxLibDir", "-lmlx",
                    // C++ stdlib
                    "-lc++",
                    // Apple frameworks
                    "-framework", "Metal",
                    "-framework", "MetalPerformanceShaders",
                    "-framework", "Accelerate",
                    "-framework", "Foundation",
                )
            }
        }
    }

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            api(kotlin("stdlib-common"))
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        val jvmMain by getting {
            dependencies {
                implementation(project(":llm-core"))
                implementation(project(":llm-inference:llama"))
                implementation(project(":llm-runtime:kllama"))
                implementation(libs.kotlinx.cli)
                implementation(libs.kotlinx.coroutines)
                implementation(libs.skainet.lang.core)
                implementation(libs.skainet.backend.cpu)
                implementation(libs.skainet.io.core)
                implementation(libs.skainet.io.gguf)
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
            }
        }

        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting
        }

        val nativeMain by creating {
            dependsOn(commonMain.get())
            dependencies {
                implementation(project(":llm-core"))
                implementation(project(":llm-inference:llama"))
                implementation(project(":llm-runtime:kllama"))
                implementation(libs.kotlinx.coroutines)
                implementation(libs.skainet.lang.core)
                implementation(libs.skainet.compile.core)
                implementation(libs.skainet.backend.cpu)
                implementation(libs.skainet.io.core)
                implementation(libs.skainet.io.gguf)
                implementation(libs.kotlinx.io.core)
            }
        }

        val macosMain by creating {
            dependsOn(nativeMain)
            dependencies {
                implementation(libs.skainet.backend.metal)
                implementation(libs.skainet.backend.mlx)
            }
        }

        val macosArm64Main by getting {
            dependsOn(macosMain)
        }

        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }
    }
}

val shadowJar by tasks.getting(com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar::class) {
    val jvmJar = tasks.named<org.gradle.jvm.tasks.Jar>("jvmJar")
    dependsOn(jvmJar)
    from(jvmJar.map { it.archiveFile })
    configurations = listOf(project.configurations.getByName("jvmRuntimeClasspath"))
    archiveBaseName.set("llm-performance")
    archiveClassifier.set("all")
    archiveVersion.set("")
    manifest {
        attributes(
            "Main-Class" to "sk.ainet.performance.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true",
        )
    }
    mergeServiceFiles()
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    minHeapSize = "2g"
    maxHeapSize = "12g"
}
