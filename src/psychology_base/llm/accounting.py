"""Shared accounting primitives; historical serial study ledgers remain compatible."""
from decimal import Decimal
import json
from decimal import ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from psychology_base.llm.execution import ExecutionDeferred, remaining_time


def token_cost(input_tokens, output_tokens, input_rate, output_rate):
    return (Decimal(str(input_rate)) * input_tokens + Decimal(str(output_rate)) * output_tokens) / 1_000_000


_USD_SCALE = Decimal(10) ** 12


def _units(amount, *, limit=False):
    value = Decimal(str(amount))
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid accounting amount")
    units = int((value * _USD_SCALE).to_integral_value(rounding=ROUND_FLOOR if limit else ROUND_CEILING))
    if units > 2 ** 62:
        raise ValueError("Accounting amount exceeds the integer ledger range")
    return units


def _utc_now():
    """Reservation-time clock; callers cannot choose an allowance period."""
    return datetime.now(timezone.utc)


class RuntimeBudget:
    """Durable admission across routers, threads and processes using SQLite.

    Reserve before dispatch. A crash, timeout, invalid/missing usage or failed
    call holds its full reservation. No automatic retries/refunds/period reset.
    This differs from the legacy serial study JSONL format deliberately: the
    original study Budget below remains its sole reader/writer contract.
    """

    def __init__(self, policy):
        policy.require_paid_allowance()
        self.policy = policy
        self.path = Path(policy.ledger).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The same allowance cannot be enlarged or re-priced by another router.
        identity = policy.to_mapping()
        identity.pop("ledger")
        encoded = json.dumps(identity, sort_keys=True, default=str)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS policies (budget TEXT PRIMARY KEY, contract TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, budget TEXT NOT NULL, session TEXT NOT NULL,
                operation TEXT NOT NULL, purpose TEXT NOT NULL, provider TEXT NOT NULL,
                model TEXT NOT NULL, kind TEXT NOT NULL, units INTEGER NOT NULL,
                state TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
                latency_ms REAL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("CREATE INDEX IF NOT EXISTS calls_scope ON calls(budget,session,operation)")
            db.execute("CREATE TABLE IF NOT EXISTS cache_hits (budget TEXT NOT NULL, session TEXT NOT NULL, "
                       "purpose TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL)")
            row = db.execute("SELECT contract FROM policies WHERE budget=?", (policy.budget_id,)).fetchone()
            if row is not None and row[0] != encoded:
                raise ExecutionDeferred("Shared budget contract differs; existing allowance is unchanged")
            db.execute("INSERT OR IGNORE INTO policies VALUES (?,?)", (policy.budget_id, encoded))
            # Additive, atomic schema migration: historical rows and contracts
            # stay intact and have no invented user attribution or period.
            columns = {row[1] for row in db.execute("PRAGMA table_info(calls)")}
            for name in ("user_id", "period_day", "period_month"):
                if name not in columns:
                    db.execute(f"ALTER TABLE calls ADD COLUMN {name} TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS calls_user_period ON calls(budget,user_id,period_month,period_day)")
            db.execute("CREATE INDEX IF NOT EXISTS calls_user_day ON calls(budget,user_id,period_day)")
            db.execute("""CREATE TABLE IF NOT EXISTS session_users (
                budget TEXT NOT NULL, session TEXT NOT NULL, user_id TEXT NOT NULL,
                PRIMARY KEY(budget,session))""")
            db.execute("""CREATE TABLE IF NOT EXISTS user_period_clocks (
                budget TEXT NOT NULL, user_id TEXT NOT NULL, latest_day TEXT NOT NULL,
                PRIMARY KEY(budget,user_id))""")

    @contextmanager
    def _connect(self):
        # Connection per transaction; no shared connection/thread affinity.
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=remaining_time(10))
            with db:
                yield db
        except sqlite3.Error as exc:
            raise ExecutionDeferred("Accounting storage unavailable; paid execution deferred") from exc
        finally:
            if db is not None:
                db.close()

    def _bind_user(self, db, session, user_id):
        if self.policy.user_period_limits is None:
            return None
        if not isinstance(user_id, str) or not user_id.strip() or user_id != user_id.strip():
            raise ExecutionDeferred("User period allowances require an explicit stable user identifier")
        if not isinstance(session, str) or not session.strip():
            raise ExecutionDeferred("User period allowances require a session identifier")
        row = db.execute("SELECT user_id FROM session_users WHERE budget=? AND session=?",
                         (self.policy.budget_id, session)).fetchone()
        if row is not None and row[0] != user_id:
            raise ExecutionDeferred("Paid session belongs to another user; identity cannot be changed")
        db.execute("INSERT OR IGNORE INTO session_users VALUES (?,?,?)",
                   (self.policy.budget_id, session, user_id))
        return user_id

    def bind_session(self, *, session, user_id):
        """Bind the caller's authenticated identity before routing or cache use."""
        if self.policy.user_period_limits is not None:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self._bind_user(db, session, user_id)

    @staticmethod
    def _periods():
        now = _utc_now()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ExecutionDeferred("Accounting requires an aware UTC clock")
        day = now.astimezone(timezone.utc).date().isoformat()
        return day, day[:7]

    def reserve(self, ceiling, *, session, operation, purpose, provider, model, kind, user_id=None):
        if not all(isinstance(value, str) and value.strip() for value in (session, operation, purpose)):
            raise ExecutionDeferred("Paid calls require session, operation and purpose")
        units = _units(ceiling)
        policy = self.policy
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            remaining_time()
            user_id = self._bind_user(db, session, user_id)
            day = month = None
            if db.execute("SELECT 1 FROM calls WHERE budget=? AND state='overrun' LIMIT 1",
                          (policy.budget_id,)).fetchone():
                raise ExecutionDeferred("Shared budget is held after a provider usage overrun")
            scopes = [
                ("budget=?", (policy.budget_id,), policy.max_calls, policy.max_usd),
                ("budget=? AND session=?", (policy.budget_id, session), policy.session_max_calls, policy.session_max_usd),
                ("budget=? AND session=? AND operation=?", (policy.budget_id, session, operation),
                 policy.operation_max_calls, policy.operation_max_usd),
            ]
            if user_id is not None:
                day, month = self._periods()
                previous = db.execute("SELECT latest_day FROM user_period_clocks WHERE budget=? AND user_id=?",
                                      (policy.budget_id, user_id)).fetchone()
                if previous is not None and day < previous[0]:
                    raise ExecutionDeferred("Accounting clock moved to an earlier day; paid execution deferred")
                limits = policy.user_period_limits
                scopes.extend([
                    ("budget=? AND user_id=? AND period_day=?", (policy.budget_id, user_id, day),
                     limits.daily_max_calls, limits.daily_max_usd),
                    ("budget=? AND user_id=? AND period_month=?", (policy.budget_id, user_id, month),
                     limits.monthly_max_calls, limits.monthly_max_usd),
                ])
            for where, args, max_calls, max_usd in scopes:
                count, spent = db.execute(f"SELECT COUNT(*),COALESCE(SUM(units),0) FROM calls WHERE {where}", args).fetchone()
                if count >= max_calls or spent + units > _units(max_usd, limit=True):
                    raise ExecutionDeferred("Shared paid call/spend allowance exhausted before dispatch")
            dispatch = uuid.uuid4().hex
            remaining_time()
            db.execute("INSERT INTO calls(id,budget,session,operation,purpose,provider,model,kind,units,state,"
                       "user_id,period_day,period_month) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (dispatch, policy.budget_id, session, operation, purpose, provider, model, kind, units,
                        "reserved", user_id, day, month))
            if user_id is not None:
                db.execute("INSERT INTO user_period_clocks VALUES (?,?,?) ON CONFLICT(budget,user_id) "
                           "DO UPDATE SET latest_day=excluded.latest_day", (policy.budget_id, user_id, day))
        return dispatch

    def settle(self, dispatch, *, actual=None, input_tokens=None, output_tokens=None, latency_ms=0, overrun=False):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT units,state FROM calls WHERE id=? AND budget=?",
                             (dispatch, self.policy.budget_id)).fetchone()
            if row is None or row[1] != "reserved":
                raise ValueError("Missing or already settled runtime reservation")
            units, state = row[0], "uncertain"
            if overrun or (actual is not None and _units(actual) > units):
                db.execute("UPDATE calls SET state='overrun',input_tokens=?,output_tokens=?,latency_ms=? WHERE id=?",
                           (input_tokens, output_tokens, latency_ms, dispatch))
                db.commit()
                raise ExecutionDeferred("Provider usage exceeded its reservation; budget review required")
            if actual is not None:
                charged = _units(actual)
                units, state = charged, "completed"
            db.execute("UPDATE calls SET units=?,state=?,input_tokens=?,output_tokens=?,latency_ms=? WHERE id=?",
                       (units, state, input_tokens, output_tokens, latency_ms, dispatch))

    def record_cache_hit(self, *, session, purpose, provider, model, kind):
        with self._connect() as db:
            db.execute("INSERT INTO cache_hits VALUES (?,?,?,?,?,?)",
                       (self.policy.budget_id, session, purpose, provider, model, kind))

    def summary(self):
        with self._connect() as db:
            count, units, uncertain = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(units),0),COALESCE(SUM(state!='completed'),0) FROM calls WHERE budget=?",
                (self.policy.budget_id,)).fetchone()
            hits = db.execute("SELECT COUNT(*) FROM cache_hits WHERE budget=?", (self.policy.budget_id,)).fetchone()[0]
        return dict(provider_attempts=count, accounted_usd=str(Decimal(units) / _USD_SCALE),
                    usage_uncertain=bool(uncertain), budget_id=self.policy.budget_id, cache_hits=hits)

    def user_summary(self, user_id):
        """Current UTC period reservations/charges, including unfinished calls.

        Read-only: inspecting a new period never resets any allowance. Cache
        hits are free and remain in the existing project-level summary.
        """
        limits = self.policy.user_period_limits
        if limits is None or not isinstance(user_id, str) or not user_id.strip() or user_id != user_id.strip():
            raise ValueError("User summary requires configured period limits and a stable user identifier")
        day, month = self._periods()
        result = dict(budget_id=self.policy.budget_id, user_id=user_id, timezone="UTC")
        with self._connect() as db:
            db.execute("BEGIN")
            for name, column, period in (("daily", "period_day", day), ("monthly", "period_month", month)):
                count, units, uncertain = db.execute(
                    f"SELECT COUNT(*),COALESCE(SUM(units),0),COALESCE(SUM(state!='completed'),0) "
                    f"FROM calls WHERE budget=? AND user_id=? AND {column}=?",
                    (self.policy.budget_id, user_id, period)).fetchone()
                result[name] = dict(period=period, provider_attempts=count,
                                    accounted_usd=str(Decimal(units) / _USD_SCALE), usage_uncertain=bool(uncertain),
                                    max_calls=getattr(limits, name + "_max_calls"),
                                    max_usd=str(getattr(limits, name + "_max_usd")))
        return result

def positive_decimal(value):
    value = Decimal(str(value))
    if not value.is_finite() or value <= 0:
        raise ValueError("Cost limits and rates must be finite and positive")
    return value


def request_input_bound(messages, *, response_schema=None):
    """Use the recorder's conservative framing bound for preparation and dispatch."""
    schema_bytes = (0 if response_schema is None else
        len(json.dumps(response_schema,ensure_ascii=False,separators=(',', ':')).encode('utf-8')) + 128)
    return sum(len(message.content.encode('utf-8')) + 64 for message in messages) + 512 + schema_bytes


class Budget:
    def __init__(self, writer, *, max_calls=4200, max_usd=Decimal("50"), local_only=False):
        if type(max_calls) is not int or not 0 < max_calls <= 4200:
            raise ValueError("Invalid provider-attempt limit")
        self.writer = writer
        self.max_calls = max_calls
        self.local_only = local_only
        self.max_usd = Decimal(str(max_usd)) if local_only else positive_decimal(max_usd)
        if local_only and self.max_usd != 0:
            raise ValueError("Local evaluation permits zero additional API spend")
        if self.max_usd > 50:
            raise ValueError("Study spend ceiling exceeds approved US$50")
        self.spent = Decimal(0)
        self.attempts = self.cache_hits = self.dispatches = 0
        self.uncertain = False
        self.pending = set()
        # All live stages share one ledger, including failed attempts. An
        # unmatched reservation remains one potentially billed attempt.
        if writer.path.exists():
            pending = {}
            for line in writer.path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                key=row['dispatch']
                if row['event']=='reservation':
                    if key in pending or type(key) is not int or key != self.dispatches+1:
                        raise ValueError('Duplicate/out-of-order call reservation')
                    amount=Decimal(row['reserved_usd']) if row.get('local_only') is True else positive_decimal(row['reserved_usd'])
                    if row.get('local_only') is True and amount != 0:
                        raise ValueError('Local call has a nonzero API reservation')
                    pending[key]=amount; self.dispatches=key; self.spent+=amount; self.attempts+=1
                elif row['event'] in ('completion','failure'):
                    if key not in pending: raise ValueError('Call event has no unique reservation')
                    amount=pending.pop(key)
                    if row['event']=='completion':
                        charge=Decimal(row['accounted_usd'])
                        if not charge.is_finite() or charge<0 or charge>amount or type(row['cached']) is not bool:
                            raise ValueError('Invalid completed-call accounting')
                        self.spent+=charge-amount
                        self.cache_hits+=int(row['cached']); self.attempts-=int(row['cached'])
                        self.uncertain |= not row['usage_valid']
                    else: self.uncertain=True
                else: raise ValueError('Unknown provider ledger event')
            self.uncertain |= bool(pending)
            self.pending = set(pending)
            if (not self.local_only and self.spent>self.max_usd) or self.attempts>self.max_calls:
                raise ValueError('Existing study ledger exceeds requested budget')

    def reserve(self, ceiling, request):
        if self.local_only and (ceiling != 0 or request.get('local_only') is not True):
            raise ValueError("Local budget rejects external calls before dispatch")
        if self.attempts >= self.max_calls or (not self.local_only and self.spent + ceiling > self.max_usd):
            raise ValueError("Provider call/spend budget exhausted before dispatch")
        self.dispatches += 1
        self.attempts += 1
        self.spent += ceiling
        self.pending.add(self.dispatches)
        record = {"event": "reservation", "dispatch": self.dispatches,
                  "reserved_usd": str(ceiling), **request}
        self.writer.write(record)
        return self.dispatches

    def summary(self):
        summary = {"dispatches": self.dispatches, "provider_attempts": self.attempts,
                "cache_hits": self.cache_hits, "accounted_usd": str(self.spent),
                "usage_uncertain": self.uncertain or bool(self.pending), "max_calls": self.max_calls,
                "max_usd": str(self.max_usd), "cost_kind": "frozen-rate token estimate or retained reservation"}
        if self.local_only:
            summary.update(local_only=True, additional_external_api_usd="0", hardware_cost_usd=None,
                           cost_kind="Historical remote estimates retained; local API billing is inapplicable; hardware cost unmeasured")
        return summary


