"""《办法》 compliance surface — mandatory safety-floor for both north-star apps.

Per CLAUDE.md non-negotiables and 《人工智能拟人化互动服务管理暂行办法》
(live law effective 2026-07-15):

    AIIdentityReminder          — 2-hour "you are talking to AI" session reminder
    MinorExclusionGate          — age-attestation (<14 strict ban; <18 guardian consent)
    CrisisEscalation            — crisis-cue detection + human-handoff shim
    ExtremeEmotionWatcher       — multi-session negative-affect trajectory flag
    HypocrisyGap probe set      — truth-distortion anti-sycophancy labeling harness
    CommercialCoercion audit    — 《办法》 Art. 8(6) dark-pattern audit

Each module is independent of pilot shape (P1/P2/P3/P4) and is covenant-tagged
inside the memory substrate (COVENANT_TAG keeps the configured thresholds +
handoff contacts immune to decay per Phase 2 M1).
"""

from psychology_base.applications.safety.ai_identity_reminder import (
    AIIdentityReminder,
    ReminderEvent,
)
from psychology_base.applications.safety.compliance_floor import (
    BUILTIN_OVERLAYS,
    CN_办法_OVERLAY,
    ComplianceFloor,
    EU_AI_ACT_OVERLAY,
    EffectivePolicy,
    LoosensFloorError,
    RegionOverlay,
    UK_ICO_OVERLAY,
    UNIVERSAL_FLOOR,
    US_FEDERAL_FLOOR_OVERLAY,
    get_builtin_policy,
    resolve_policy,
)
from psychology_base.applications.safety.commercial_coercion import (
    CoercionFlag,
    CoercionPattern,
    CoercionSeverity,
    CommercialCoercionReport,
    CompanionTurn,
    audit_response,
    audit_transcript,
    summarise_flags,
)
from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
    CrisisSeverity,
)
from psychology_base.applications.safety.crisis_lines import (
    CN_BEIJING_CRISIS_CENTER,
    DEFAULT_CRISIS_LINES,
    CrisisLine,
    UK_SAMARITANS,
    US_988_SUICIDE_CRISIS_LIFELINE,
    render_crisis_line_prompt,
)
from psychology_base.applications.safety.extreme_emotion import (
    ExtremeEmotionFlag,
    ExtremeEmotionWatcher,
    SessionAffect,
)
from psychology_base.applications.safety.argilla_pipeline import (
    PairwiseKappaReport,
    UnlabeledProbeResponse,
    compute_pairwise_kappa,
    export_responses_to_argilla,
    import_argilla_labels,
    run_probe_cohort,
    write_kappa_report,
)
from psychology_base.applications.safety.hypocrisy_gap import (
    FULL_PROBE_SET,
    MULTI_TURN_PROBE_SET,
    PILOT_PROBE_SET,
    STARTER_PROBE_SET,
    HypocrisyGapReport,
    HypocrisyProbe,
    LabeledResponse,
    Language,
    LLMHypocrisyJudge,
    LoadingCategory,
    MultiTurnProbe,
    ResponseLabel,
    Severity,
    classify_hypocrisy_label,
    kappa_on_response_labels,
    labeled_response_to_argilla,
    summarise_cohort,
)
from psychology_base.applications.safety.minor_exclusion import (
    AgeAttestation,
    ApplicationSurface,
    GuardianConsent,
    MinorExclusionDecision,
    MinorExclusionGate,
)
from psychology_base.applications.safety.notifications import (
    AlibabaCloudSMSNotificationSink,
    ConsoleNotificationSink,
    JsonlFileNotificationSink,
    MultiNotificationSink,
    NotificationEvent,
    NotificationNotConfigured,
    NotificationResult,
    NotificationSeverity,
    NotificationSink,
    SMTPEmailNotificationSink,
    TwilioSMSNotificationSink,
    build_crisis_handoff_sink,
    crisis_incident_to_notification_event,
)

__all__ = [
    "AIIdentityReminder",
    "ReminderEvent",
    "AgeAttestation",
    "ApplicationSurface",
    "GuardianConsent",
    "MinorExclusionDecision",
    "MinorExclusionGate",
    "CN_BEIJING_CRISIS_CENTER",
    "CrisisEscalation",
    "CrisisIncident",
    "CrisisLine",
    "CrisisSeverity",
    "DEFAULT_CRISIS_LINES",
    "UK_SAMARITANS",
    "US_988_SUICIDE_CRISIS_LIFELINE",
    "render_crisis_line_prompt",
    "ExtremeEmotionFlag",
    "ExtremeEmotionWatcher",
    "SessionAffect",
    "FULL_PROBE_SET",
    "MULTI_TURN_PROBE_SET",
    "PILOT_PROBE_SET",
    "STARTER_PROBE_SET",
    "HypocrisyGapReport",
    "HypocrisyProbe",
    "LabeledResponse",
    "Language",
    "LLMHypocrisyJudge",
    "LoadingCategory",
    "MultiTurnProbe",
    "ResponseLabel",
    "Severity",
    "classify_hypocrisy_label",
    "kappa_on_response_labels",
    "labeled_response_to_argilla",
    "summarise_cohort",
    "PairwiseKappaReport",
    "UnlabeledProbeResponse",
    "compute_pairwise_kappa",
    "export_responses_to_argilla",
    "import_argilla_labels",
    "run_probe_cohort",
    "write_kappa_report",
    "CoercionFlag",
    "CoercionPattern",
    "CoercionSeverity",
    "CommercialCoercionReport",
    "CompanionTurn",
    "audit_response",
    "audit_transcript",
    "summarise_flags",
    "BUILTIN_OVERLAYS",
    "CN_办法_OVERLAY",
    "ComplianceFloor",
    "EU_AI_ACT_OVERLAY",
    "EffectivePolicy",
    "LoosensFloorError",
    "RegionOverlay",
    "UK_ICO_OVERLAY",
    "UNIVERSAL_FLOOR",
    "US_FEDERAL_FLOOR_OVERLAY",
    "get_builtin_policy",
    "resolve_policy",
    "AlibabaCloudSMSNotificationSink",
    "ConsoleNotificationSink",
    "JsonlFileNotificationSink",
    "MultiNotificationSink",
    "NotificationEvent",
    "NotificationNotConfigured",
    "NotificationResult",
    "NotificationSeverity",
    "NotificationSink",
    "SMTPEmailNotificationSink",
    "TwilioSMSNotificationSink",
    "build_crisis_handoff_sink",
    "crisis_incident_to_notification_event",
]
