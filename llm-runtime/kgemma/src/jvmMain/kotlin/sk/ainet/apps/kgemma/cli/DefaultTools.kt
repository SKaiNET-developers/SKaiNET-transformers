package sk.ainet.apps.kgemma.cli

import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.cli.CalculatorTool
import sk.ainet.apps.kllama.cli.ListFilesTool

/**
 * Named built-in tools available to `kgemma --agent --tools=…`.
 *
 * Keeps the CLI surface small: adding a new tool is one entry here, and the
 * `--tools` flag parser can validate names against [names] to fail fast on
 * typos.
 */
internal object DefaultTools {

    private val factories: Map<String, () -> Tool> = linkedMapOf(
        "calculator" to ::CalculatorTool,
        "list_files" to ::ListFilesTool
    )

    val names: Set<String> = factories.keys

    fun byName(name: String): Tool? = factories[name]?.invoke()

    /**
     * Parse a comma-separated list of tool names and return the instantiated
     * tools in the order given. Unknown names are reported through [onUnknown].
     * Returns `null` if any name is unknown (caller should surface usage).
     */
    fun parse(spec: String?, onUnknown: (String) -> Unit): List<Tool>? {
        if (spec.isNullOrBlank()) return emptyList()
        val tokens = spec.split(',').map { it.trim() }.filter { it.isNotEmpty() }
        val result = mutableListOf<Tool>()
        var ok = true
        for (name in tokens) {
            val tool = byName(name)
            if (tool == null) {
                onUnknown(name)
                ok = false
            } else {
                result += tool
            }
        }
        return if (ok) result else null
    }
}
