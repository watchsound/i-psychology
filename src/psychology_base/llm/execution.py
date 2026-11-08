"""Execution permission is independent of model quality tier.

The research mode preserves historical caller/ledger contracts. Application
factories always select local or paid explicitly; paid permission requires a
separate, durable allowance and a named operation, never a failed local probe.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal
import math
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from concurrent.futures import Future, TimeoutError as FutureTimeout
from threading import BoundedSemaphore, Event, Lock, Thread
import time


# Explicit offline local diagnostics may take longer on CPU. Serving limits
# remain separate and the constructor-only opt-in cannot come from config.
MAX_OFFLINE_LOCAL_TIMEOUT_S = 3600


@dataclass(frozen=True)
class InferenceLimit:
    """Attempts and conservative input-plus-output token bounds, not billing."""
    max_calls: int
    max_tokens: int

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.max_calls, self.max_tokens)):
            raise ValueError("Inference limits require nonnegative integers")
        if (self.max_calls == 0) != (self.max_tokens == 0):
            raise ValueError("Disabled inference requires both limits to be zero")


@dataclass(frozen=True)
class ConversationLimits:
    """Independent allowances preserve room for the user-facing response.

    These do not authorize paid transport. ExecutionPolicy and its durable
    ledger still govern permission and money. Research outside a serving turn
    retains its existing experiment contract.
    """
    response: InferenceLimit = field(default_factory=lambda: InferenceLimit(1, 16384))
    auxiliary: InferenceLimit = field(default_factory=lambda: InferenceLimit(0, 0))
    non_generation: InferenceLimit = field(default_factory=lambda: InferenceLimit(8, 65536))

    def __post_init__(self):
        if any(type(getattr(self, name)) is not InferenceLimit
               for name in ("response", "auxiliary", "non_generation")):
            raise ValueError("Each conversation allowance must be an InferenceLimit")

    @classmethod
    def from_mapping(cls, raw):
        names = {"response", "auxiliary", "non_generation"}
        if not isinstance(raw, dict) or set(raw) - names:
            raise ValueError("Unknown conversation allowance or non-mapping configuration")
        values = {}
        for name, value in raw.items():
            if not isinstance(value, dict) or set(value) != {"max_calls", "max_tokens"}:
                raise ValueError("Conversation allowances require max_calls and max_tokens")
            values[name] = InferenceLimit(**value)
        return cls(**values)


class ConversationBudget:
    """One in-process turn, shared by nested operations and copied contexts.

    Every admitted attempt retains its full token bound, including failures
    and abandoned transports. Actual billing remains in RuntimeBudget. Counts
    cover inference dispatches; model verification/metadata are not inference.
    """

    def __init__(self, limits: ConversationLimits):
        if type(limits) is not ConversationLimits:
            raise ValueError("Conversation budget requires a ConversationLimits profile")
        self.limits = limits
        self._lock = Lock()
        self._closed = False
        self._blocked = False
        self._usage = {name: dict(attempts=0, completed=0, failed=0,
                                 token_bound=0, cache_hits=0, denied=0)
                       for name in ("response", "auxiliary", "non_generation")}

    @staticmethod
    def bucket(kind, role):
        return role if kind == "chat" else "non_generation"

    def reserve(self, kind, role, input_bound, output_bound):
        name = self.bucket(kind, role)
        if name not in self._usage:
            raise ValueError("Unknown conversation inference role")
        if (type(input_bound) is not int or input_bound <= 0
                or type(output_bound) is not int or output_bound < 0
                or (kind == "chat" and output_bound == 0)):
            raise ExecutionDeferred("Serving inference requires finite input/output bounds")
        bound = input_bound + output_bound
        with self._lock:
            usage, limit = self._usage[name], getattr(self.limits, name)
            if (self._closed or self._blocked or usage["attempts"] >= limit.max_calls
                    or usage["token_bound"] + bound > limit.max_tokens):
                usage["denied"] += 1
                raise ExecutionDeferred(f"Conversation {name} inference allowance exhausted")
            usage["attempts"] += 1
            usage["token_bound"] += bound
        return name

    def finish(self, name, *, success, overrun=False):
        with self._lock:
            self._usage[name]["completed" if success else "failed"] += 1
            self._blocked |= overrun

    def check_capacity(self, kind, role, *, calls, token_bound):
        """Refuse a known oversized plan before dispatch; reserve still owns admission.

        This check does not reserve or promise completion against concurrent
        work. Every physical dispatch must still use the same reserve method.
        """
        name = self.bucket(kind, role)
        if (name not in self._usage or any(type(n) is not int or n < 0 for n in (calls, token_bound))
                or (calls == 0) != (token_bound == 0)):
            raise ValueError('Inference plans require bounded calls and tokens')
        with self._lock:
            usage, limit = self._usage[name], getattr(self.limits, name)
            if (self._closed or calls > 0 and (self._blocked or usage['attempts'] + calls > limit.max_calls
                    or usage['token_bound'] + token_bound > limit.max_tokens)):
                usage['denied'] += 1
                raise ExecutionDeferred(f'Conversation {name} inference plan exceeds the remaining allowance')

    def cache_hit(self, kind, role):
        with self._lock:
            self._usage[self.bucket(kind, role)]["cache_hits"] += 1

    def close(self):
        with self._lock:
            self._closed = True

    def hold(self):
        with self._lock:
            self._blocked = True

    def snapshot(self):
        with self._lock:
            return dict(closed=self._closed, blocked=self._blocked,
                        limits={name: dict(max_calls=getattr(self.limits, name).max_calls,
                                           max_tokens=getattr(self.limits, name).max_tokens)
                                for name in self._usage},
                        usage={name: dict(value) for name, value in self._usage.items()})


@dataclass(frozen=True)
class Operation:
    id: str
    purpose: str
    deadline: float | None
    cancelled: Event = field(default_factory=Event, compare=False)
    conversation: ConversationBudget | None = field(default=None, compare=False)
    inference_role: str = field(default="auxiliary", compare=False)


current_operation = ContextVar("llm_operation", default=None)
_request_slots = {kind: BoundedSemaphore(4) for kind in ("local", "paid")}


def remaining_time(timeout=None):
    """Bound each transport/admission wait by the same monotonic deadline."""
    operation = current_operation.get()
    if operation is None or operation.deadline is None:
        return timeout
    remaining = operation.deadline - time.monotonic()
    if remaining <= 0 or operation.cancelled.is_set():
        raise ExecutionDeferred("Operation deadline exhausted")
    return remaining if timeout is None else min(timeout, remaining)


@contextmanager
def request_slot(transport="local"):
    """Bound abandoned transports without waiting for them on caller exit.

    Synchronous SDK/socket cancellation cannot prove a request was unbilled.
    Late results are discarded; a timed-out worker owns its slot until it exits.
    Each transport class has four slots. Stuck paid calls cannot occupy the
    local model's capacity; the process-wide maximum is eight active workers.
    """
    timeout = remaining_time()
    if timeout is None:
        yield lambda invoke: invoke()
        return
    slots = _request_slots[transport]
    if not slots.acquire(timeout=timeout):
        raise ExecutionDeferred("No model transport slot before the operation deadline")
    started = False

    def send(invoke):
        nonlocal started
        remaining_time()
        context, future = copy_context(), Future()

        def work():
            try:
                def run():
                    remaining_time()
                    return invoke()
                future.set_result(context.run(run))
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                slots.release()

        worker = Thread(target=work, name=f"bounded-{transport}-llm-request", daemon=True)
        worker.start()
        started = True
        try:
            result = future.result(timeout=remaining_time())
            remaining_time()
            return result
        except FutureTimeout as exc:
            # Future.result also re-raises a provider's own TimeoutError.
            # Preserve that failure reason when its request ended before the
            # operation deadline; it is not an abandoned transport worker.
            if future.done():
                remaining_time()
                return future.result()
            current_operation.get().cancelled.set()
            raise ExecutionDeferred("Model request exceeded the operation deadline; late result discarded") from exc

    try:
        remaining_time()
        yield send
    finally:
        if not started:
            slots.release()


class ExecutionDeferred(RuntimeError):
    """No answer was generated; callers may preserve input and defer the turn."""


@dataclass(frozen=True)
class PaidRate:
    """Explicit frozen rates and bounds, supplied by the budget owner.

    No built-in prices or serving allowance: a model name is not authorization.
    """
    input_per_million: str
    output_per_million: str
    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self):
        for value in (self.input_per_million, self.output_per_million):
            number = Decimal(str(value))
            if not number.is_finite() or number < 0:
                raise ValueError("Rates must be finite and nonnegative")
        for value in (self.max_input_tokens, self.max_output_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("Token bounds must be nonnegative integers")
        if self.max_input_tokens == 0:
            raise ValueError("A positive input bound is required")


@dataclass(frozen=True)
class UserPeriodLimits:
    """Explicit paid allowances per stable user and UTC calendar day/month.

    Period rollover never resets project/session/operation allowances. These
    limits do not authenticate a user or grant paid transport permission.
    """
    daily_max_calls: int
    daily_max_usd: str
    monthly_max_calls: int
    monthly_max_usd: str

    def __post_init__(self):
        for value in (self.daily_max_calls, self.monthly_max_calls):
            if type(value) is not int or value <= 0:
                raise ValueError("User period call limits must be positive integers")
        for value in (self.daily_max_usd, self.monthly_max_usd):
            number = Decimal(str(value))
            if not number.is_finite() or number <= 0:
                raise ValueError("User period spend limits must be finite and positive")


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str = "local"
    ledger: Path | None = None
    budget_id: str = ""
    max_calls: int = 0
    max_usd: str = "0"
    session_max_calls: int = 0
    session_max_usd: str = "0"
    operation_max_calls: int = 0
    operation_max_usd: str = "0"
    request_timeout_s: float = 60.0
    operation_timeout_s: float = 60.0
    # Keys: provider/kind/model, e.g. tongyi/chat/<explicit model>.
    rates: dict[str, PaidRate] = field(default_factory=dict)
    # Constructor-only opt-in for bounded offline work. It is not a serving
    # configuration field and does not change serialized paid-ledger identities.
    offline_local_diagnostic: InitVar[bool] = False
    user_period_limits: UserPeriodLimits | None = None

    def __post_init__(self, offline_local_diagnostic):
        if self.mode not in ("research", "local", "paid"):
            raise ValueError("Execution mode must be research/local/paid")
        if (type(offline_local_diagnostic) is not bool
                or offline_local_diagnostic and self.mode != "local"):
            raise ValueError("Offline diagnostic deadlines require an explicit local-only opt-in")
        timeout_limit = MAX_OFFLINE_LOCAL_TIMEOUT_S if offline_local_diagnostic else 300
        for value in (self.request_timeout_s, self.operation_timeout_s):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= timeout_limit:
                raise ValueError(f"Execution timeouts must be within (0, {timeout_limit}] seconds")
        for value in (self.max_calls, self.session_max_calls, self.operation_max_calls):
            if type(value) is not int or value < 0:
                raise ValueError("Call limits must be nonnegative integers")
        for value in (self.max_usd, self.session_max_usd, self.operation_max_usd):
            number = Decimal(str(value))
            if not number.is_finite() or number < 0:
                raise ValueError("Spend limits must be finite and nonnegative")
        if any(not isinstance(value, PaidRate) for value in self.rates.values()):
            raise ValueError("Execution rates must be PaidRate records")
        if self.user_period_limits is not None and type(self.user_period_limits) is not UserPeriodLimits:
            raise ValueError("User period limits must be a UserPeriodLimits record")

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("execution must be a mapping")
        values = dict(raw)
        if values.get("offline_local_diagnostic", False) is not False:
            raise ValueError("Offline diagnostic deadlines require programmatic opt-in, not serving configuration")
        values["rates"] = {key: PaidRate(**value) for key, value in values.get("rates", {}).items()}
        if values.get("user_period_limits") is not None:
            limits = values["user_period_limits"]
            fields = {"daily_max_calls", "daily_max_usd", "monthly_max_calls", "monthly_max_usd"}
            if not isinstance(limits, dict) or set(limits) != fields:
                raise ValueError("User period limits require explicit daily/monthly call and spend caps")
            values["user_period_limits"] = UserPeriodLimits(**limits)
        if values.get("ledger") is not None:
            values["ledger"] = Path(values["ledger"])
        return cls(**values)

    def require_paid_allowance(self):
        if (self.mode != "paid" or self.ledger is None or not self.budget_id.strip()
                or not self.rates or min(self.max_calls, self.session_max_calls, self.operation_max_calls) <= 0
                or min(Decimal(str(self.max_usd)), Decimal(str(self.session_max_usd)),
                       Decimal(str(self.operation_max_usd))) <= 0):
            raise ExecutionDeferred("Paid execution requires an explicit shared budget, rates and limits")

    def to_mapping(self):
        """Serialized policy contract; absent period limits retain legacy bytes."""
        from dataclasses import asdict
        values = asdict(self)
        if self.user_period_limits is None:
            values.pop("user_period_limits")
        return values
