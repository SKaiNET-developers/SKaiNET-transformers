plugins {
    alias(libs.plugins.skainet.multiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Generic ASR task types (Transcription, DecodingOptions, FeatureFrames) -- moved up from the
// asr-whisper-iree-cartridge downstream repo (its asr-domain module), where they lived only
// because that's where the original whisper-cli extraction happened to put them, not because they're Whisper-specific. Both the Whisper and
// Moonshine cartridge families depend on these; keeping them in one cartridge family's repo made
// the other family structurally dependent on it for generic plumbing. Deliberately
// dependency-free (no SKaiNET engine deps) so any downstream consumer can take it without pulling
// in a specific backend.
//
// Targets: gradle.properties (skainet.targets) -- must be readable while the plugin applies, see
// SkainetTargets' own doc comment for why this can't live in the skainet { } block below.
// kotlin-test in commonTest is added automatically (SkainetMultiplatformExtension's
// kotlinTestInCommonTest default).
skainet {
    // sk.ainet.multiplatform 1.1.0 defaults the JVM target to 17; the engine artifacts and the rest of
    // this build are JVM 21, and inlining across the two fails to compile.
    jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21
    namespace = "sk.ainet.asr.domain"
}
