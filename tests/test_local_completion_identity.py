"""Provider identity is fixed by its transport, not a mutable display name."""
import pytest
from types import SimpleNamespace

from psychology_base.applications.measurement.local_query_binding import _validate_local_completion, LocalChatMembershipAssessor
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.types import Completion, TaskTier


def test_mutable_ollama_name_cannot_make_a_different_provider_identity_valid():
    provider=OllamaProvider(name='different-provider',advanced_model='declared-fixture',local_only=True,
        expected_digest='a'*64,context_length=8192,seed=0,json_mode=True)
    completion=Completion(content='{"labels":["match"]}',provider='different-provider',model_id='declared-fixture',
        tier=TaskTier.ADVANCED,prompt_tokens=100,completion_tokens=12,finish_reason='stop',
        metadata=dict(local_only=True,done=True,response_model='declared-fixture',
            model_attestation=dict(digest='a'*64),context_length=8192,seed=0,json_mode=True))
    with pytest.raises(ValueError):_validate_local_completion(completion,provider,64)


def test_noncanonical_ollama_name_is_rejected_before_constructing_an_assessor():
    provider=OllamaProvider(name='different-provider',advanced_model='declared-fixture',local_only=True,
        expected_digest='a'*64,context_length=8192,seed=0,json_mode=True)
    router=SimpleNamespace(execution=SimpleNamespace(mode='local'),providers={'ollama':provider})
    with pytest.raises(ValueError):LocalChatMembershipAssessor(router)


def test_provider_name_mutation_is_detected_before_any_request():
    provider=OllamaProvider(advanced_model='declared-fixture',local_only=True,
        expected_digest='a'*64,context_length=8192,seed=0,json_mode=True)
    router=SimpleNamespace(execution=SimpleNamespace(mode='local'),providers={'ollama':provider})
    assessor=LocalChatMembershipAssessor(router)
    provider.name='different-provider'
    with pytest.raises(ValueError,match='configuration changed'):assessor([])


def test_paired_clock_includes_preparation_in_final_write_bound():
    from psychology_base.eval.benchmarks.local_membership_audit import paired_final_write_wall_s
    started=dict(monotonic=100.,wall_ns=1034_000_000_000)
    anchor=dict(monotonic=140.,wall_ns=1040_000_000_000)
    final=1803_000_000_000
    assert (final-started['wall_ns'])/1e9==769
    assert paired_final_write_wall_s(started,anchor,final)==803


@pytest.mark.parametrize('anchor,final',[
    (dict(monotonic=True,wall_ns=1000),2000),
    (dict(monotonic=90,wall_ns=1000),2000),
    (dict(monotonic=float('nan'),wall_ns=1000),2000),
    (dict(monotonic=110,wall_ns=True),2000),
    (dict(monotonic=110,wall_ns=2000),1000),
])
def test_invalid_timing_evidence_cannot_pass_a_corrected_audit(anchor,final):
    from psychology_base.eval.benchmarks.local_membership_audit import paired_final_write_wall_s
    with pytest.raises(ValueError):paired_final_write_wall_s(dict(monotonic=100.),anchor,final)
