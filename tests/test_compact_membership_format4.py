"""Adapter compatibility with explicit score fixtures, not model accuracy."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from psychology_base.applications.measurement.semantic_membership import LocalMembershipAssessor, membership_inputs
from psychology_base.llm.execution import ExecutionDeferred
from tests.test_membership_context import sample


def router():
    calls = []

    def classify(pairs):
        calls.append(pairs)
        return [dict(match=.9, nonmatch=.05, uncertain=.05) for _ in pairs]

    return SimpleNamespace(execution=SimpleNamespace(mode='local'),
        local_reranker=SimpleNamespace(model_id='authored-software-test', tokenizer_sha256='fixed-test-tokenizer'),
        classify_pairs=classify, operation=lambda *args, **kwargs: nullcontext()), calls


def test_format4_adapter_preserves_shared_projection_constraints_and_one_batch():
    value, calls = router()
    case = sample()
    case['records'][1]['speaker'] = None
    case['records'][2]['speaker'] = 'tutor'
    rows = membership_inputs(case, format_version=4)
    assessor = LocalMembershipAssessor(value, format_version=4)
    labels = assessor(rows)
    assert labels == [r['constraint'] or 'match' for r in rows]
    assert calls == [[r['pair'] for r in rows if r['constraint'] is None]]
    assert len({LocalMembershipAssessor(value, format_version=v).identity for v in (2, 3, 4)}) == 3
    assert LocalMembershipAssessor(value).format_version == 3


@pytest.mark.parametrize('version,expected', [
    (2, '5b5f0a5a30003283e61efb186a32c50bb282fb46caac56915890f91233f4b152'),
    (3, 'e0c44484ce4887c2cd642e1dfffb436a678a336e38b47469da87f87025762532'),
])
def test_legacy_assessor_identities_stay_exact(version, expected):
    value, calls = router()
    assert LocalMembershipAssessor(value, format_version=version).identity == expected
    assert calls == []


@pytest.mark.parametrize('configured, supplied', [(4, 2), (4, 3), (2, 4), (3, 4)])
def test_explicit_format4_cannot_be_interpreted_under_another_adapter_contract(configured, supplied):
    value, calls = router()
    with pytest.raises(ValueError, match='pinned input format'):
        LocalMembershipAssessor(value, format_version=configured)(membership_inputs(sample(), format_version=supplied))
    assert calls == []


def test_mixed_versions_and_mutated_format_fail_before_local_dispatch():
    value, calls = router()
    assessor = LocalMembershipAssessor(value, format_version=4)
    rows = membership_inputs(sample(), format_version=4)
    rows[0].pop('input_format_version')
    with pytest.raises(ValueError, match='pinned input format'): assessor(rows)
    assessor.format_version = 3
    with pytest.raises(ExecutionDeferred): assessor(membership_inputs(sample(), format_version=4))
    assert calls == []


def test_numeric_format_marker_is_an_integer_contract():
    value, calls = router()
    rows = membership_inputs(sample(), format_version=4)
    rows[0]['input_format_version'] = 4.0
    with pytest.raises(ValueError, match='pinned input format'):
        LocalMembershipAssessor(value, format_version=4)(rows)
    assert calls == []


def test_format4_compact_adapter_shares_one_assessment_with_selection_and_binding():
    from psychology_base.applications.context import render_context_block
    from psychology_base.applications.llm_responders import request_context_selection
    from psychology_base.applications.reporting_order import ReportingMembershipSession, bind_reporting_order
    from psychology_base.companion.reconstruction import ordered_selection
    from tests.test_reporting_membership_session import fragments, QUERY

    value, calls = router()

    def software_scores(pairs):
        calls.append(pairs)
        # Explicit cells keyed by fixture text, never by fragment/hash order.
        # This is a transport oracle, not an act-recognition algorithm.
        assert len(pairs) == 16
        positives = {('A question?', 'Description: the  question'), ('An answer.', 'Description: the  answer')}
        return [dict(match=.9, nonmatch=.05, uncertain=.05)
                if (json.loads(premise)['focus_utterance']['text'], hypothesis.splitlines()[0]) in positives else
                dict(match=.05, nonmatch=.9, uncertain=.05) for premise, hypothesis in pairs]

    value.classify_pairs = software_scores
    assessor = LocalMembershipAssessor(value, format_version=4)
    session = ReportingMembershipSession('tutor', QUERY, assessor)
    policy = request_context_selection('Policy.', QUERY, application='tutor', format_version=4,
        max_input=8000, mode='occurrences', membership_session=session)
    fs = fragments()
    selected, trace = ordered_selection(fs, {f.id: 0 for f in fs}, 6, policy)
    context = render_context_block([f for f, _ in selected], assistant_speaker='tutor', format_version=4)
    binding = bind_reporting_order('tutor', QUERY, context, selection_trace=trace, membership_session=session)
    assert binding.selected is not None and binding.selected.source_position == 0
    assert binding.resolution.query_target == 1 and len(calls) == 1


@pytest.mark.parametrize('version', [True, 0, 1, 5, '4', None])
def test_unsupported_compact_versions_stay_rejected(version):
    with pytest.raises(ValueError): LocalMembershipAssessor(router()[0], format_version=version)
