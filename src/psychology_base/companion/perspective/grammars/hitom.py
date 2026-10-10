"""HiToM corpus grammar — 15-container MC + nested-belief chain scaffold.

Migrated from ``higher_order_tom_scorer.py:235-338`` as part of the Phase
9+ perspective substrate consolidation. The scorer dispatches on
``item.conversation.metadata["source"] == "hitom"`` and delegates to the
builders here.

**Why corpus-specific grammar.** HiToM items ship a 15-container
multiple-choice option set per item plus a nested-belief chain in the
question ("A thinks B thinks ..."). The generic FANToM prompt shells
drop the choices and frame questions as first-order. Running HiToM
through those shells forces free-form container-token generation out of
a 15-way vocabulary of underscored names and loses the nested-belief
framing — which is the Pillar E v2 architectural protocol fix that
lifted the gate from v1 FAIL (0.590/0.530/+0.060) to v2 PASS
(0.806/0.430/+0.376 at order 2).

**Both paths see choices.** HiToM paper standard eval protocol. Only the
context block differs:
- Perspective path: witness-filtered co-presence intersection (built at
  the routing layer via :class:`CopresenceScope`).
- Bare path: full story — the LLM has to track the object through 15+
  events including events chain agents couldn't have observed.

These are protocol-conformance primitives on top of the architectural
witness filter, not architecture themselves.
"""

from __future__ import annotations

from psychology_base.eval.benchmarks.public_datasets.fantom import FANToMItem


def hitom_choices_block(item: FANToMItem) -> str:
    """Render the HiToM 15-container multiple-choice block."""

    choices = item.question.choices or ()
    if not choices:
        return ""
    bullet_lines = [f"- {c}" for c in choices]
    return "Possible answers (pick exactly one):\n" + "\n".join(bullet_lines)


def hitom_chain_scaffold(item: FANToMItem) -> str:
    """Explain the nested-belief chain to the LLM.

    Order 0: factual question (world truth).
    Order 1: single-agent belief (first-order ToM).
    Order >= 2: nested belief — explicitly describe the belief chain so
    the LLM frames the question as "A's model of B's belief" rather than
    "A's own belief", even though for symmetric chains the intersection
    semantics are identical.
    """

    metadata = item.conversation.metadata or {}
    chain = tuple(metadata.get("agent_chain", []) or [])
    order = item.question.tom_order or 0

    if order == 0 or not chain:
        return (
            "This is a factual question about where the object actually is "
            "in the world."
        )
    if order == 1:
        return (
            f"This is a first-order belief question about what {chain[0]} "
            f"believes. {chain[0]} can only know what {chain[0]} directly "
            f"observed in the same room."
        )
    inner = chain[-1]
    outer_chain = chain[:-1]
    desc = f"{outer_chain[0]}'s model of "
    for ag in outer_chain[1:]:
        desc += f"{ag}'s model of "
    desc += f"{inner}'s belief"
    return (
        f"This is an order-{order} nested-belief question. You are reasoning "
        f"about {desc}. For a belief to be in {outer_chain[0]}'s model of "
        f"what {inner} knows, every agent in the chain "
        f"({', '.join(chain)}) must have been in the same room when the "
        f"event occurred — otherwise one of them is modeling an event they "
        f"couldn't have observed through the others."
    )


def build_hitom_perspective_prompt(item: FANToMItem, mentalistic_ctx: str) -> str:
    """Prompt shell for the HiToM perspective (mentalistic) path.

    Builds a chain-aware prompt around the witness-filtered events. Both
    the choices and the chain scaffolding are included so the LLM has
    the HiToM paper's standard evaluation surface, not a stripped-down
    FANToM-shaped question.
    """

    scaffold = hitom_chain_scaffold(item)
    context_label = (
        "Events in the belief chain's co-presence intersection "
        "(all chain agents were in the same room at these moments):"
    )
    body = f"{context_label}\n{mentalistic_ctx or '(no events in the intersection)'}"
    choices_block = hitom_choices_block(item)
    question_block = f"Question: {item.question.question_text}"
    answer_hint = (
        "Answer with the exact container name from the list above "
        "(for example: green_bathtub). Do not add any other words."
    )
    blocks = [scaffold, body]
    if choices_block:
        blocks.append(choices_block)
    blocks.extend([question_block, answer_hint])
    return "\n\n".join(blocks)


def build_hitom_bare_prompt(item: FANToMItem) -> str:
    """Prompt shell for the HiToM bare-LLM baseline.

    Full story + chain scaffolding + choices. The LLM has to track the
    object location through the full event stream — including events
    that occurred after chain agents exited — and reason about which
    moves the chain agents could and could not have witnessed.
    """

    scaffold = hitom_chain_scaffold(item)
    story = item.conversation.context[0].utterance if item.conversation.context else ""
    body = f"Full story:\n{story or '(empty)'}"
    choices_block = hitom_choices_block(item)
    question_block = f"Question: {item.question.question_text}"
    answer_hint = (
        "Answer with the exact container name from the list above "
        "(for example: green_bathtub). Do not add any other words."
    )
    blocks = [scaffold, body]
    if choices_block:
        blocks.append(choices_block)
    blocks.extend([question_block, answer_hint])
    return "\n\n".join(blocks)


__all__ = [
    "build_hitom_bare_prompt",
    "build_hitom_perspective_prompt",
    "hitom_chain_scaffold",
    "hitom_choices_block",
]
