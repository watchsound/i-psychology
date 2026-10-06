from __future__ import annotations

from typing import Optional

from psychology_base.eval.adapters import Fragment, Observation, Recall


class EchoMemory:
    """Null baseline — no reconstruction, no selection, no mentalistic structure, no time.

    Observations are appended verbatim. recall() does case-insensitive substring match
    on the cue, returning the most recent matching observation's content. tick() is a
    no-op. inspect() exposes the raw observation list so fragment-count-based metrics
    can run against the baseline — setting the floor against which V-track thresholds
    (currently TBD in docs/sota-goal.md) get calibrated.
    """

    def __init__(self) -> None:
        self._observations: list[Observation] = []

    def observe(self, event: Observation) -> None:
        self._observations.append(event)

    def recall(self, cue: str, context: Optional[dict] = None) -> Recall:
        needle = cue.lower()
        matches = [o for o in self._observations if needle in o.content.lower()]
        content = matches[-1].content if matches else ""
        fragments = tuple(
            Fragment(
                id=f"echo-{idx}",
                content=obs.content,
                timestamp=obs.timestamp,
                affect=dict(obs.affect),
                tags=obs.tags,
            )
            for idx, obs in enumerate(matches)
        )
        return Recall(
            cue=cue,
            content=content,
            fragments_used=fragments,
            metadata={"match_count": len(matches)},
        )

    def tick(self) -> None:
        return None

    def inspect(self) -> dict:
        return {
            "fragment_count": len(self._observations),
            "fragments": [
                {
                    "id": f"echo-{idx}",
                    "timestamp": obs.timestamp.isoformat(),
                    "speaker": obs.speaker,
                    "tags": list(obs.tags),
                }
                for idx, obs in enumerate(self._observations)
            ],
        }
