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
    api(project(":llm-api"))
    api(project(":llm-core"))
    api(project(":llm-agent"))
    api(project(":llm-inference:bert"))

    implementation(libs.skainet.lang.core)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.kotlinx.serialization.json)

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
