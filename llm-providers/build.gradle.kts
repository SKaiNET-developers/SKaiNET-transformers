import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
        freeCompilerArgs.add("-Xjvm-default=all")
    }
    explicitApi()
}

dependencies {
    implementation(platform(project(":llm-bom")))

    api(project(":llm-api"))
    api(project(":llm-core"))
    api(project(":llm-agent"))
    api(project(":llm-inference:bert"))

    implementation(libs.skainet.lang.core)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.kotlinx.serialization.json)

    // BertEmbeddingModel one-call factory: local SafeTensors loading + built-in
    // Hugging Face download (hf:// URIs). kotlinx-io appears in data-source's
    // public API but is declared implementation upstream, so add it explicitly.
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.safetensors)
    implementation(libs.skainet.data.source)
    implementation(libs.kotlinx.io.core)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

// Pick up Java sources alongside Kotlin in src/test
sourceSets["test"].java.srcDir("src/test/java")

tasks.test {
    useJUnitPlatform()
}
