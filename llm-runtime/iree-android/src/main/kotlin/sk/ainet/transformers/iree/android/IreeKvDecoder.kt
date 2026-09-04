package sk.ainet.transformers.iree.android

/**
 * Greedy tool-call decoding over an [IreeKvSession]: prefill the catalog prefix once
 * ([prefillPrefix] → [IreeKvSession.Snapshot]), then per turn [generate] restores the snapshot,
 * feeds the utterance in chunk calls and decodes until `eos`/`maxNewTokens`.
 *
 * Timing of the last [generate] call is exposed in [lastTiming] (ms), so a smoke app or a
 * cartridge can log restore / chunk / decode separately.
 */
public class IreeKvDecoder(public val session: IreeKvSession, public val prefillSeq: Int) {

    public class Timing(public val restoreMs: Long, public val chunkMs: Long, public val chunkCalls: Int, public val decodeMs: Long, public val decodeTokens: Int) {
        public val totalMs: Long get() = restoreMs + chunkMs + decodeMs
    }
    public var lastTiming: Timing? = null
        private set

    /** Prefill [prefixIds] (must be < [prefillSeq]) once and return a snapshot of the cache after it. */
    public fun prefillPrefix(prefixIds: IntArray): IreeKvSession.Snapshot {
        require(prefixIds.size in 1 until prefillSeq) { "prefix ${prefixIds.size} must be in 1..${prefillSeq - 1}" }
        val padded = IntArray(prefillSeq).also { prefixIds.copyInto(it) }
        session.prefill(padded, prefixIds.size)
        return session.snapshot()
    }

    /**
     * Restore [prefix], run [utteranceIds] through the chunk graph (several calls if longer than
     * `spec.chunk`), then decode greedily. Returns the generated ids (without the eos token).
     */
    public fun generate(prefix: IreeKvSession.Snapshot, utteranceIds: IntArray, eosTokenId: Int, maxNewTokens: Int): IntArray {
        require(utteranceIds.isNotEmpty()) { "empty utterance" }
        val c = session.spec.chunk
        var t = System.nanoTime()
        session.restore(prefix)
        val restoreMs = (System.nanoTime() - t) / 1_000_000
        t = System.nanoTime()
        var next = -1
        var calls = 0
        var off = 0
        while (off < utteranceIds.size) {
            val n = minOf(c, utteranceIds.size - off)
            val buf = IntArray(c).also { utteranceIds.copyInto(it, 0, off, off + n) }
            next = session.chunk(buf, n)
            calls++
            off += n
        }
        val chunkMs = (System.nanoTime() - t) / 1_000_000
        t = System.nanoTime()
        val out = ArrayList<Int>(maxNewTokens)
        var produced = 0
        while (produced < maxNewTokens) {
            if (next == eosTokenId) break
            out += next
            produced++
            if (produced >= maxNewTokens) break
            next = session.step(next)
        }
        val decodeMs = (System.nanoTime() - t) / 1_000_000
        lastTiming = Timing(restoreMs, chunkMs, calls, decodeMs, produced)
        return out.toIntArray()
    }
}
