// Generic Android JNI runtime for the DSL -> StableHLO -> IREE compiled path
// (transformers#305). Plain com.android.library — NOT Kotlin Multiplatform — matching the
// engine repo's skainet-backends/skainet-backend-jni-cpu precedent, the one proven shape
// in this codebase for "JNI/.so + jniLibs bundling". No kotlin("android") plugin needed:
// AGP ships built-in Kotlin support.
//
// This module knows nothing about any specific model: IreeRedecodeSession's native shim
// drives ANY vmfb that follows the redecode graph contract (tensor<1xSEQxi32> ->
// tensor<SEQxi32>, in-graph argMax already applied, weights external via a caller-supplied
// `.irpa`). :llm-inference:smollm2 is its first producer of a (vmfb, irpa, function-name)
// triple — see that module's docs/smollm2-vmfb.md for a concrete example.
plugins {
    // AGP 9 ships built-in Kotlin support — applying kotlin("android") is an
    // error since 9.0 (see skainet-backend-jni-cpu in the engine repo, the
    // precedent this module follows), so this module uses the android plugin alone.
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

android {
    namespace = "sk.ainet.transformers.iree.android"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }

    androidResources {
        // vmfb/irpa assets (bundled by consumers, not this module) are dense binaries;
        // compression buys nothing and slows the first-run copy to filesDir.
        noCompress += listOf("vmfb", "irpa")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
        }
    }

    // Publishing variant selection is configured by the vanniktech
    // maven-publish plugin (AndroidSingleVariantLibrary "release") — same as
    // skainet-backend-jni-cpu; declaring singleVariant() again here conflicts.
}

dependencies {
    testImplementation(libs.kotlin.test)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}
