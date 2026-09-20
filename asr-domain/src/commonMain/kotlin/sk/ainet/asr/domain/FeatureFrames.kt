package sk.ainet.asr.domain

/**
 * Neutral, framework-free container for 2-D features (row-major `[rows][cols]`, flattened).
 * Mel spectrograms and other feature matrices cross module boundaries as this type so the
 * domain never imports a tensor library.
 */
public class FeatureFrames(
    public val rows: Int,
    public val cols: Int,
    public val data: FloatArray,
) {
    init {
        require(data.size == rows * cols) { "FeatureFrames size ${data.size} != $rows*$cols" }
    }
}
