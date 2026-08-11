plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Reusable Gemma-on-IREE runtime: the on-device decode side of the
// DSL -> StableHLO -> IREE path. Given a compiled vmfb + its .irpa weights +
// the matching GGUF (tokenizer), it runs the greedy decode loop and parses the
// compact tool-call output. Pairs with the gemma3 -> StableHLO export in
// :llm-inference:gemma (host) that produces the vmfb this module consumes.
//
//   commonMain  — CompactCodec + ToolCall (the tool-call grammar) + the
//                 FunctionGemma ToolCallingSupport/ParserStrategy/ChatTemplate
//                 bridge into llm-agent's provider architecture (#35/#36)
//   native      — IreeRuntime (drives iree-run-module) + GemmaDecoder
kotlin {
    jvm()
    linuxX64()
    linuxArm64()
    macosArm64() // Apple Silicon host dev (mirrors :llm-runtime:kgemma); uses the same nativeMain sources

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            // api, not implementation: FunctionGemmaToolCallingSupport / the
            // parser strategy / chat template expose llm-agent's
            // ToolCallingSupport, ChatTemplate and ToolCall in public
            // signatures. Direction is agent <- gemma-iree (not the reverse):
            // llm-agent publishes js/wasm/android/ios targets this module
            // does not have, so the provider lives HERE and is registered at
            // runtime via ToolCallingSupportResolver.register(...).
            api(project(":llm-agent"))
            // llm-agent keeps kotlinx-serialization-json `implementation`-scoped;
            // ToolCall.arguments is a JsonObject, so we need our own dependency
            // to construct one.
            implementation(libs.kotlinx.serialization.json)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.serialization.json)
        }

        // Native (board + host) gets the IREE driver + decode loop. The SL2610
        // ships only a statically-linked iree-run-module (no libiree C API), so
        // IreeRuntime drives it as a posix subprocess; GemmaDecoder needs the
        // GGUF tokenizer from :llm-core and kotlinx-io for file access. nativeMain
        // is the default-hierarchy parent of linux{X64,Arm64}Main.
        nativeMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(project(":llm-core"))
            implementation(libs.kotlinx.io.core)
        }
    }
}
