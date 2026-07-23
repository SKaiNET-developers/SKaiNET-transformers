package sk.ainet.apps.kllama.java

import sk.ainet.apps.llm.PrefillStrategy

/**
 * Configuration for text generation, used by KLlamaJava facades.
 *
 * Example usage from Java:
 * ```java
 * GenerationConfig config = GenerationConfig.builder()
 *     .maxTokens(256)
 *     .temperature(0.7f)
 *     .batchedPrefill(64) // opt-in: 3-10x faster prompt ingestion
 *     .build();
 * ```
 */
public class GenerationConfig private constructor(
    public val maxTokens: Int,
    public val temperature: Float,
    public val prefillStrategy: PrefillStrategy
) {
    public companion object {
        @JvmStatic
        public fun builder(): Builder = Builder()

        /** Default configuration: 256 max tokens, temperature 0.8, autoregressive prefill. */
        @JvmStatic
        public fun defaults(): GenerationConfig = Builder().build()
    }

    public class Builder {
        private var maxTokens: Int = 256
        private var temperature: Float = 0.8f
        private var prefillStrategy: PrefillStrategy = PrefillStrategy.Autoregressive

        public fun maxTokens(maxTokens: Int): Builder {
            this.maxTokens = maxTokens
            return this
        }

        public fun temperature(temperature: Float): Builder {
            this.temperature = temperature
            return this
        }

        /**
         * Ingest the prompt via chunked `forwardBatched` calls instead of one
         * `forward` per token — typically 3–10× faster prefill on long
         * prompts, numerically equivalent (see the prefill equivalence tests).
         */
        @JvmOverloads
        public fun batchedPrefill(maxBatch: Int = 64): Builder {
            this.prefillStrategy = PrefillStrategy.Batched(maxBatch)
            return this
        }

        /** Set an explicit [PrefillStrategy] (Kotlin-friendly variant). */
        public fun prefillStrategy(strategy: PrefillStrategy): Builder {
            this.prefillStrategy = strategy
            return this
        }

        public fun build(): GenerationConfig = GenerationConfig(maxTokens, temperature, prefillStrategy)
    }
}
