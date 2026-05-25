plugins {
    alias(libs.plugins.kotlinMultiplatform) apply false
    alias(libs.plugins.androidMultiplatformLibrary) apply false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.kotlinSerialization) apply false
    alias(libs.plugins.shadow) apply false
}

// Require JDK 21+ for bytecode target; JDK 25 recommended (set via jenv local 25.0).
// Produces Java 21 bytecode via --release / jvmTarget for backward compatibility.
subprojects {
    require(JavaVersion.current() >= JavaVersion.VERSION_21) {
        "This project requires JDK 21+, but found ${JavaVersion.current()}"
    }

    // Kotlin Multiplatform projects – set jvmTarget on every JVM-like target
    plugins.withId("org.jetbrains.kotlin.multiplatform") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension::class.java)?.apply {
            targets.withType(org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget::class.java) {
                compilerOptions {
                    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
                }
            }
        }
    }
    // Kotlin/JVM projects
    plugins.withId("org.jetbrains.kotlin.jvm") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension::class.java)?.apply {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
            }
        }
    }

    // Java sources – produce Java 21 bytecode regardless of the JDK used to compile.
    afterEvaluate {
        if (!plugins.hasPlugin("com.android.library") && !plugins.hasPlugin("com.android.application") && !plugins.hasPlugin("com.android.kotlin.multiplatform.library")) {
            tasks.withType<JavaCompile>().configureEach {
                options.release.set(21)
            }
        }
    }

    tasks.withType<Test>().configureEach {
        maxHeapSize = "8192m"
        useJUnitPlatform {
            // -PsmokeReference: narrow to the 3 reference smoke tests
            // (Qwen3 / Gemma-4 / BERT+LEAF). Implies @Tag("smoke-reference").
            // Pair with -PincludeIntegration when the models are present.
            if (project.hasProperty("smokeReference")) {
                includeTags("smoke-reference")
            } else if (!project.hasProperty("includeIntegration")) {
                excludeTags("integration")
            }
        }
    }
}
