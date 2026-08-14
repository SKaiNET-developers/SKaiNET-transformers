package sk.ainet.apps.kgemma.cli.android

import sk.ainet.apps.kgemma.NativeFunctionGemma

/**
 * Real (non-stub) native CLI entry point, androidNativeArm32-only — unlike the shared
 * `sk.ainet.apps.kgemma.cli.main` stub (nativeMain, used by linux/macos), this one
 * actually runs [NativeFunctionGemma]. Own package so both can coexist without a
 * redeclaration clash (Main.kt's `main` would otherwise collide with nativeMain's).
 *
 * Usage: kgemma <gguf-path> <instruction text>
 */
public fun main(args: Array<String>) {
    val gguf = args.getOrNull(0)
        ?: run { println("usage: kgemma <gguf-path> <instruction text>"); return }
    val instruction = args.drop(1).joinToString(" ")
    if (instruction.isBlank()) {
        println("usage: kgemma <gguf-path> <instruction text>")
        return
    }

    val fg = NativeFunctionGemma.fromGguf(gguf)
    val turn = fg.call(instruction)
    println("text: ${turn.text}")
    turn.calls.forEach { println("call: ${it.tool}(${it.args})") }
}
