package sk.ainet.transformers.java;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import sk.ainet.apps.kllama.java.GenerationConfig;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;
import sk.ainet.models.bert.java.KBertJava;
import sk.ainet.models.bert.java.KBertSession;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Fast smoke test that exercises both consumer surfaces in one JVM:
 *   - LEAF (mdbr-leaf-mt) BERT embeddings via {@link KBertJava}
 *   - Llama 3 chat generation via {@link KLlamaJava}
 *
 * Deliberately does not drive a full agent loop: tool-calling round-trips
 * with a 1B-param model on CPU prefill the prompt 1–2k tokens per round,
 * which pushes a "smoke" run into the 10+ minute range. The tool-calling
 * surface is already covered by {@link KLlamaJavaToolCallingTest}.
 *
 * Gated on env vars / cache fallbacks so CI without the checkpoints skips:
 *   LLAMA3_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct-Q*.gguf
 *   LEAF_MODEL_DIR=/path/to/MongoDB_mdbr-leaf-ir/   (model.safetensors + vocab.txt)
 */
class Llama3LeafSmokeTest {

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    void leafEmbeddingAndLlama3GenerationBothWork() throws Exception {
        Path leafDir = resolveLeafModelDir();
        Path llamaPath = resolveLlama3Path();
        assumeTrue(leafDir != null,
                "No LEAF model dir found — set LEAF_MODEL_DIR or place mdbr-leaf-mt under ~/.deliverance/MongoDB_mdbr-leaf-ir/.");
        assumeTrue(llamaPath != null,
                "No Llama 3 GGUF found — set LLAMA3_MODEL_PATH or place a Llama-3.2-Instruct GGUF in the HF cache.");

        // --- LEAF: embed three strings and check that paraphrases score higher
        //           against each other than against an unrelated topic.
        try (KBertSession bert = KBertJava.loadSafeTensors(leafDir)) {
            float[] embA = bert.encode("How do I reset my password?");
            float[] embB = bert.encode("What is the procedure to recover account access?");
            float[] embC = bert.encode("The Pacific Ocean is the largest body of water on Earth.");

            assertNotNull(embA, "LEAF returned null embedding for A");
            assertTrue(embA.length > 0, "LEAF returned empty embedding for A");
            assertTrue(embA.length == embB.length && embB.length == embC.length,
                    "LEAF embedding dimensions inconsistent: " + embA.length + "/" + embB.length + "/" + embC.length);

            float simParaphrase = cosineSimilarity(embA, embB);
            float simUnrelated = cosineSimilarity(embA, embC);
            assertTrue(simParaphrase > simUnrelated,
                    "Expected paraphrase similarity (" + simParaphrase
                            + ") to exceed unrelated similarity (" + simUnrelated + ")");
        }

        // --- Llama 3: generate a tiny continuation. We don't constrain the text
        //           (1B model is noisy) — only that the call completes and emits
        //           non-empty output within the token budget.
        try (KLlamaSession session = KLlamaJava.loadGGUF(llamaPath, /* systemPrompt */ null)) {
            GenerationConfig cfg = GenerationConfig.builder()
                    .maxTokens(16)
                    .temperature(0f) // greedy: deterministic + skips sampling overhead
                    .build();

            String response = session.generate("The capital of Slovakia is", cfg);

            assertNotNull(response, "Llama 3 generate() returned null");
            assertTrue(!response.isBlank(),
                    "Llama 3 generate() returned blank/empty text");
        }
    }

    private static Path resolveLlama3Path() {
        String env = System.getenv("LLAMA3_MODEL_PATH");
        if (env != null && !env.isBlank()) {
            Path p = Path.of(env);
            return Files.exists(p) ? p : null;
        }
        Path snapshotsDir = Path.of(System.getProperty("user.home"),
                ".cache", "huggingface", "hub",
                "models--bartowski--Llama-3.2-1B-Instruct-GGUF", "snapshots");
        if (!Files.isDirectory(snapshotsDir)) return null;
        File[] snapshots = snapshotsDir.toFile().listFiles(File::isDirectory);
        if (snapshots == null) return null;
        for (String name : new String[]{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q8_0.gguf"}) {
            for (File s : snapshots) {
                Path candidate = s.toPath().resolve(name);
                if (Files.exists(candidate)) return candidate;
            }
        }
        return null;
    }

    private static Path resolveLeafModelDir() {
        String env = System.getenv("LEAF_MODEL_DIR");
        if (env != null && !env.isBlank()) {
            Path p = Path.of(env);
            return Files.isDirectory(p) ? p : null;
        }
        Path deliverance = Path.of(System.getProperty("user.home"),
                ".deliverance", "MongoDB_mdbr-leaf-ir");
        if (Files.isDirectory(deliverance)) return deliverance;
        return null;
    }

    private static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Embedding dimensions must match: "
                    + a.length + " vs " + b.length);
        }
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        if (denom < 1e-12) return 0f;
        return (float) (dot / denom);
    }
}
