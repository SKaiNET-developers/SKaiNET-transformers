package sk.ainet.transformers.java;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
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
 * Reference smoke test #3 — BERT encoder + LEAF retrieval (kbert runner).
 *
 * Locked in as part of the SKaiNET 0.25.0 bump. Exercises:
 *  - SafeTensors load path for BERT-family encoders via {@link KBertJava}.
 *    With SKaiNET 0.25.0's new {@code DTypePolicy} surface the SafeTensors
 *    loader is policy-aware; this test stays on the adaptive default
 *    (FP32 dequant) and pins the consumer Java API end-to-end.
 *  - Cosine-similarity sanity check: paraphrases of "reset my password" must
 *    embed closer to each other than to an unrelated topic. Catches silent
 *    embedding regressions the way the existing
 *    {@link Llama3LeafSmokeTest} does for the LEAF-only path.
 *  - Java consumer surface (no Kotlin glue) — proves the published Java API
 *    contract still resolves against 0.25.0.
 *
 * Tagged {@code @Tag("smoke-reference")} so it runs only under
 * {@code ./gradlew test -PsmokeReference -PincludeIntegration}. Self-skips
 * via {@link org.junit.jupiter.api.Assumptions#assumeTrue} when the LEAF
 * checkpoint is not available, so CI without artifacts stays green.
 *
 * Locator chain (first match wins):
 *  1. {@code LEAF_MODEL_DIR} env var
 *  2. {@code ~/.deliverance/MongoDB_mdbr-leaf-ir/}
 *  3. {@code ~/.cache/huggingface/hub/models--MongoDB--mdbr-leaf-ir/snapshots/<sha>/}
 */
@Tag("smoke-reference")
@Tag("integration")
class BertLeafReferenceSmokeTest {

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    void leafEncoderProducesParaphraseAwareEmbeddings() throws Exception {
        Path leafDir = resolveLeafModelDir();
        assumeTrue(leafDir != null,
                "No LEAF model dir found — set LEAF_MODEL_DIR or place mdbr-leaf-ir " +
                        "under ~/.deliverance/MongoDB_mdbr-leaf-ir/.");

        try (KBertSession bert = KBertJava.loadSafeTensors(leafDir)) {
            float[] embA = bert.encode("How do I reset my password?");
            float[] embB = bert.encode("What is the procedure to recover account access?");
            float[] embC = bert.encode("The Pacific Ocean is the largest body of water on Earth.");

            assertNotNull(embA, "LEAF returned null embedding for A");
            assertTrue(embA.length > 0, "LEAF returned empty embedding for A");
            assertTrue(embA.length == embB.length && embB.length == embC.length,
                    "LEAF embedding dimensions inconsistent: " +
                            embA.length + "/" + embB.length + "/" + embC.length);

            float simParaphrase = cosineSimilarity(embA, embB);
            float simUnrelated = cosineSimilarity(embA, embC);
            System.out.println(
                    "[smoke-reference] LEAF sim(paraphrase)=" + simParaphrase
                            + " sim(unrelated)=" + simUnrelated);
            assertTrue(simParaphrase > simUnrelated,
                    "Expected paraphrase similarity (" + simParaphrase
                            + ") to exceed unrelated similarity (" + simUnrelated + ")");
        }
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

        Path snapshots = Path.of(System.getProperty("user.home"),
                ".cache", "huggingface", "hub",
                "models--MongoDB--mdbr-leaf-ir", "snapshots");
        if (Files.isDirectory(snapshots)) {
            File[] children = snapshots.toFile().listFiles(File::isDirectory);
            if (children != null) {
                for (File c : children) {
                    Path candidate = c.toPath();
                    if (Files.exists(candidate.resolve("model.safetensors"))) return candidate;
                }
            }
        }
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
