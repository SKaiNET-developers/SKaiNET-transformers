package sk.ainet.models.gemma

/**
 * Type of attention layer in Gemma 3n.
 */
public enum class LayerType {
    /** Local sliding-window attention */
    SLIDING,
    /** Global full-context attention */
    GLOBAL
}
