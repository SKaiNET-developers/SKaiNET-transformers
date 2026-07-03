package sk.ainet.models.moonshine

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Torq-**target** graph-lowering pass — rewrites each `scaledDotProductAttention`
 * node into a subgraph of **standard ops** (reshape / slice / transpose / matmul /
 * softmax / concatenate) shaped the way the Torq NPU compiler accepts:
 *   - fold `[1,H,S,D]` → `[H,S,D]` (Torq rejects 4D-batched matmul);
 *   - transpose K so QK^T is a standard `A[M,K]·B[K,N]` matmul; and
 *   - split heads into groups of ≤[maxHeadsPerTile] (>4 heads overflow NPU SRAM).
 *
 * The output is still **portable StableHLO** (it runs on llvm-cpu too) — this pass is
 * target-aware but the shared IR emitter and the model definition stay HW-agnostic.
 * It lives OUTSIDE core and plugs into `TargetOptimizers` for the `"torq"` target, so
 * the compiler core carries no Torq knowledge.
 */
class TorqAttentionTilingPass(private val maxHeadsPerTile: Int = 4) : GraphOptimizationPass {
    override val name: String = "torq-attention-tiling"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val sdpas = graph.nodes.filter { it.operationName.lowercase() == "scaleddotproductattention" }
        if (sdpas.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        val newGraph = DefaultComputeGraph()
        // Carry over every non-SDPA node unchanged.
        val sdpaIds = sdpas.map { it.id }.toSet()
        val kept = graph.nodes.filter { it.id !in sdpaIds }
        for (n in kept) newGraph.addNode(n)

        // Producer of each (nodeId, inputIndex): (source node, source output index).
        val producerOf = HashMap<Pair<String, Int>, Pair<GraphNode, Int>>()
        for (e in graph.edges) producerOf[e.destination.id to e.destinationInputIndex] = e.source to e.sourceOutputIndex
        // Consumers of each SDPA output.
        val consumersOf = graph.edges.filter { it.source.id in sdpaIds }.groupBy { it.source.id }

        for (sdpa in sdpas) {
            val q = sdpa.inputs[0]; val k = sdpa.inputs[1]; val v = sdpa.inputs[2]
            val out = sdpa.outputs[0]
            val elem = out.dtype
            // Expect [1, H, S, D].
            val h = q.shape!![1]; val sq = q.shape!![2]; val d = q.shape!![3]; val sk = k.shape!![2]
            val scale = (sdpa.operation.parameters["scale"] as? Number)?.toFloat()
                ?: (1.0f / kotlin.math.sqrt(d.toFloat()))
            val base = sdpa.id
            var counter = 0
            fun spec(vararg dims: Int) = TensorSpec("${base}_t${counter++}", dims.toList(), elem)
            val nodes = mutableListOf<GraphNode>()
            val edges = mutableListOf<Triple<Pair<GraphNode, Int>, GraphNode, Int>>() // (src,outIdx)->(dst,inIdx)
            fun op(opName: String, params: Map<String, Any>, ins: List<TensorSpec>, o: TensorSpec): GraphNode {
                val n = GraphNode("${base}_${opName}_${counter++}", GenericOperation(opName, params), ins, listOf(o))
                nodes += n; return n
            }

            // fold [1,H,S,D] -> [H,S,D]
            val q3s = spec(h, sq, d); val k3s = spec(h, sk, d); val v3s = spec(h, sk, d)
            val q3 = op("reshape", emptyMap(), listOf(q), q3s)
            val k3 = op("reshape", emptyMap(), listOf(k), k3s)
            val v3 = op("reshape", emptyMap(), listOf(v), v3s)
            // Wire the three reshapes to the SDPA's original q/k/v producers.
            producerOf[sdpa.id to 0]?.let { edges += Triple(it, q3, 0) }
            producerOf[sdpa.id to 1]?.let { edges += Triple(it, k3, 0) }
            producerOf[sdpa.id to 2]?.let { edges += Triple(it, v3, 0) }

            val groupOuts = mutableListOf<Pair<GraphNode, Int>>() // (node, headCount)
            var s = 0
            while (s < h) {
                val e = minOf(s + maxHeadsPerTile, h); val g = e - s
                fun sliceP(start: Int, limit: Int, s1: Int, s2: Int) = mapOf(
                    "start_indices" to listOf(start, 0, 0),
                    "limit_indices" to listOf(limit, s1, s2),
                    "strides" to listOf(1, 1, 1),
                )
                val qgS = spec(g, sq, d); val kgS = spec(g, sk, d); val vgS = spec(g, sk, d)
                val qg = op("slice", sliceP(s, e, sq, d), listOf(q3s), qgS).also { edges += Triple(q3 to 0, it, 0) }
                val kg = op("slice", sliceP(s, e, sk, d), listOf(k3s), kgS).also { edges += Triple(k3 to 0, it, 0) }
                val vg = op("slice", sliceP(s, e, sk, d), listOf(v3s), vgS).also { edges += Triple(v3 to 0, it, 0) }
                // transpose K [g,Sk,D] -> [g,D,Sk]
                val ktS = spec(g, d, sk)
                val kt = op("transpose", mapOf("permutation" to listOf(0, 2, 1)), listOf(kgS), ktS)
                    .also { edges += Triple(kg to 0, it, 0) }
                // scores = Qg @ Kt  (matmul -> torq-friendly batching[0] contracting[2]x[1])
                val scS = spec(g, sq, sk)
                val sc = op("matmul", emptyMap(), listOf(qgS, ktS), scS)
                    .also { edges += Triple(qg to 0, it, 0); edges += Triple(kt to 0, it, 1) }
                // scale: multiply scores by a full-shape splat of 1/sqrt(head_dim)
                // (matches convertSdpa; a rank-0 scalar mis-prints as tensor<xbf16>).
                val scaleCS = spec(g, sq, sk)
                val scaleC = op("splat_constant", mapOf("value" to scale), emptyList(), scaleCS)
                val scdS = spec(g, sq, sk)
                val scd = op("multiply", emptyMap(), listOf(scS, scaleCS), scdS)
                    .also { edges += Triple(sc to 0, it, 0); edges += Triple(scaleC to 0, it, 1) }
                // softmax over last dim
                val atS = spec(g, sq, sk)
                val at = op("softmax", mapOf("axis" to 2), listOf(scdS), atS)
                    .also { edges += Triple(scd to 0, it, 0) }
                // out = attn @ Vg
                val ogS = spec(g, sq, d)
                val og = op("matmul", emptyMap(), listOf(atS, vgS), ogS)
                    .also { edges += Triple(at to 0, it, 0); edges += Triple(vg to 0, it, 1) }
                groupOuts += og to g
                s = e
            }

            // concat groups along heads -> [H,Sq,D], then reshape -> [1,H,Sq,D]
            val catS = spec(h, sq, d)
            val cat = op("concat", mapOf("dim" to 0), groupOuts.map { it.first.outputs[0] }, catS)
            groupOuts.forEachIndexed { i, (n, _) -> edges += Triple(n to 0, cat, i) }
            val fin = op("reshape", emptyMap(), listOf(catS), out.copy(name = "${base}_out"))
            edges += Triple(cat to 0, fin, 0)

            // Commit nodes + internal edges.
            for (n in nodes) newGraph.addNode(n)
            for ((src, dst, inIdx) in edges) {
                newGraph.addEdge(
                    GraphEdge("e_${src.first.id}_${src.second}__${dst.id}_$inIdx", src.first, dst, src.second, inIdx, dst.inputs[inIdx]),
                )
            }
            // Rewire the SDPA's consumers to the final reshape.
            for (c in consumersOf[sdpa.id].orEmpty()) {
                newGraph.addEdge(c.copy(id = "${c.id}_torq", source = fin, sourceOutputIndex = 0))
            }
        }

        // Re-add all edges that don't touch an SDPA node.
        for (e in graph.edges) {
            if (e.source.id in sdpaIds || e.destination.id in sdpaIds) continue
            edgeSafeAdd(newGraph, e)
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }

    private fun edgeSafeAdd(g: DefaultComputeGraph, e: GraphEdge) {
        val byId = g.nodes.associateBy { it.id }
        val s = byId[e.source.id] ?: return
        val d = byId[e.destination.id] ?: return
        g.addEdge(e.copy(source = s, destination = d))
    }
}
