package sk.ainet.decisions.runtime

import kotlinx.serialization.json.JsonElement
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.DecisionResult
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.dsl.DecisionSchema

/** A session for the model the schema names; the schema must name one. */
public fun DecisionRuntime.openSession(schema: DecisionSchema): DecisionSession =
    openSession(requireNotNull(schema.model) { "schema '${schema.id}' names no model" })

/** Decides the schema's questions about [state] under the schema's policy unless [acceptance] overrides it. */
public fun DecisionSession.decide(
    schema: DecisionSchema,
    requestId: String,
    state: DecisionState,
    acceptance: AcceptancePolicy? = schema.policy,
): DecisionResult = decide(schema.request(requestId, state), acceptance)

public fun DecisionSession.decide(schema: DecisionSchema, requestId: String, state: String, acceptance: AcceptancePolicy? = schema.policy): DecisionResult =
    decide(schema, requestId, DecisionState.Text(state), acceptance)

public fun DecisionSession.decide(schema: DecisionSchema, requestId: String, state: JsonElement, acceptance: AcceptancePolicy? = schema.policy): DecisionResult =
    decide(schema, requestId, DecisionState.Structured(state), acceptance)
