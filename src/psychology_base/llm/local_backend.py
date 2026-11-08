"""Pinned installed assets and an owned, offline, CPU-only diagnostic process.

No process, socket, model loading or network operation occurs at import time.
GGUF inspection reads bounded metadata, never tensors. This module does not
compose model prompts or decide membership labels.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess

from psychology_base.llm.execution import remaining_time


MAX_NATIVE_OUTPUT_TOKENS = 4096


def local_memory_admission(plan, residency, available_bytes, *, separate_process=False):
    """One admission calculation; separate processes never reuse resident memory."""
    limits = plan['limits']
    cold, warm = limits['cold_memory_minimum_bytes'], limits['resident_memory_minimum_bytes']
    if (type(separate_process) is not bool or type(available_bytes) is not int or available_bytes < 0
            or type(cold) is not int or type(warm) is not int or not 0 < warm <= cold
            or type(residency) is not dict or type(residency.get('models')) is not list):
        raise ValueError('Invalid local memory admission evidence or bounds')
    candidates = [m for m in residency['models'] if type(m) is dict and m.get('name') == plan['model']]
    reused = (not separate_process and len(candidates) == 1 and candidates[0].get('digest') == plan['digest']
        and type(candidates[0].get('context_length')) is int
        and candidates[0]['context_length'] == limits['context_length']
        and type(candidates[0].get('size')) is int and candidates[0]['size'] > 0)
    required = warm if reused else cold
    return dict(available_bytes=available_bytes, required_bytes=required,
                exact_model_context_resident=reused, admitted=available_bytes >= required)


def digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a lowercase SHA-256 digest')
    return value


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    sha256: str
    signature: tuple

    def check(self):
        if self.path.is_symlink() or _stat(self.path) != self.signature:
            raise ValueError('Verified local artifact changed: ' + str(self.path))

    def record(self):
        return dict(path=str(self.path), sha256=self.sha256, bytes=self.signature[0])


def verify_file(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular installed artifact is required')
    path = path.resolve()
    before = _stat(path)
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            remaining_time()
            value.update(block)
    actual = value.hexdigest()
    if _stat(path) != before or expected is not None and actual != digest(expected):
        raise ValueError('Installed artifact digest changed: ' + str(path))
    return VerifiedFile(path, actual, before)


def inspect_gguf(path):
    """Read only the v3 metadata needed to attest this diagnostic's template.

    Limits apply to skipped values too; malformed lengths cannot seek into
    tensors, allocate arbitrary arrays, or silently read a truncated header.
    """
    widths = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    limit = min(Path(path).stat().st_size, 32 * 1024 * 1024)
    fields, keys, tokens = {}, set(), {}
    with Path(path).open('rb') as stream:
        def read(n):
            if type(n) is not int or n < 0 or stream.tell() + n > limit:
                raise ValueError('GGUF metadata exceeds its bounded header')
            value = stream.read(n)
            if len(value) != n:
                raise ValueError('Truncated GGUF metadata')
            return value

        def number(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize(fmt)))[0]

        def string():
            return read(number('Q')).decode('utf-8')

        def skip(kind):
            if kind in widths:
                read(widths[kind])
            elif kind == 8:
                read(number('Q'))
            elif kind == 9:
                subtype, count = number('I'), number('Q')
                if count > 1_000_000 or subtype not in (*widths, 8):
                    raise ValueError('Unsupported or oversized GGUF array')
                if subtype in widths:
                    read(widths[subtype] * count)
                else:
                    for _ in range(count):
                        skip(subtype)
            else:
                raise ValueError('Unknown GGUF metadata type')

        if read(4) != b'GGUF' or number('I') != 3:
            raise ValueError('Expected a little-endian GGUF v3 model')
        tensors, count = number('Q'), number('Q')
        if not 0 < tensors <= 100_000 or not 0 < count <= 4096:
            raise ValueError('Invalid GGUF header counts')
        for _ in range(count):
            key, kind = string(), number('I')
            if not key or len(key) > 1024 or key in keys:
                raise ValueError('Invalid or duplicate GGUF metadata key')
            keys.add(key)
            if key == 'tokenizer.ggml.tokens':
                if kind != 9 or number('I') != 8:
                    raise ValueError('GGUF vocabulary must contain strings')
                n = number('Q')
                if not 0 < n <= 1_000_000:
                    raise ValueError('Invalid GGUF vocabulary size')
                fields['token_count'] = n
                for i in range(n):
                    token = string()
                    if token in ('<think>', '</think>', '<|im_start|>', '<|im_end|>'):
                        if token in tokens:
                            raise ValueError('Duplicate thinking or chat token')
                        tokens[token] = i
            elif key in ('general.architecture', 'tokenizer.chat_template'):
                if kind != 8:
                    raise ValueError('Expected string GGUF model metadata')
                fields[key] = string()
            elif key.endswith('.context_length') or key == 'tokenizer.ggml.eos_token_id':
                if kind != 4:
                    raise ValueError('Expected unsigned GGUF model metadata')
                fields[key] = number('I')
            else:
                skip(kind)
        fields.update(special_tokens=tokens, header_bytes=stream.tell())
    template = fields.get('tokenizer.chat_template')
    if (fields.get('general.architecture') != 'qwen35' or type(template) is not str
            or not 0 < len(template.encode()) <= 65536
            or set(tokens) != {'<think>', '</think>', '<|im_start|>', '<|im_end|>'}
            or fields.get('tokenizer.ggml.eos_token_id') != tokens['<|im_end|>']
            or type(fields.get('qwen35.context_length')) is not int
            or fields['qwen35.context_length'] <= 0
            or '<think>' not in template or '</think>' not in template or 'enable_thinking' not in template):
        raise ValueError('GGUF lacks the explicit Qwen3.5 thinking/template contract')
    fields['template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    return fields


@dataclass(frozen=True)
class NativeModel:
    model_id: str
    source_digest: str
    weight: VerifiedFile
    source: tuple[VerifiedFile, ...]
    template: str
    template_sha256: str
    thinking_start: int
    thinking_end: int
    eos: int
    token_count: int
    training_context: int

    def check(self):
        for pin in (*self.source, self.weight):
            pin.check()

    def record(self):
        return dict(model_id=self.model_id, source_digest=self.source_digest,
            weight=self.weight.record(), source=[p.record() for p in self.source],
            template_sha256=self.template_sha256, thinking_start=self.thinking_start,
            thinking_end=self.thinking_end, eos=self.eos, token_count=self.token_count,
            training_context=self.training_context)


def installed_ollama_model(models_root, manifest, *, model_id, source_digest):
    """Resolve the exact selected manifest; content names alone are not hashes."""
    models_root = Path(models_root).resolve()
    manifest = Path(manifest).resolve()
    if not manifest.is_relative_to(models_root / 'manifests') or not model_id.strip():
        raise ValueError('Model manifest must belong to the installed model root')
    source_digest = digest(source_digest)
    root = verify_file(manifest)
    root_data = json.loads(root.path.read_text(encoding='utf-8'))
    references = [r.get('digest') for r in root_data.get('layers', []) + root_data.get('manifests', [])]
    if root.sha256 != source_digest and 'sha256:' + source_digest not in references:
        raise ValueError('Selected model digest is absent from the installed manifest')
    selected = root if root.sha256 == source_digest else verify_file(
        models_root / 'blobs' / ('sha256-' + source_digest), source_digest)
    data = json.loads(selected.path.read_text(encoding='utf-8'))
    layers = [r for r in data['layers'] if r.get('mediaType') == 'application/vnd.ollama.image.model']
    if len(layers) != 1 or type(layers[0].get('digest')) is not str or not layers[0]['digest'].startswith('sha256:'):
        raise ValueError('Expected exactly one installed GGUF weight layer')
    weight_digest = digest(layers[0]['digest'][7:])
    weight = verify_file(models_root / 'blobs' / ('sha256-' + weight_digest), weight_digest)
    if type(layers[0].get('size')) is not int or weight.signature[0] != layers[0]['size']:
        raise ValueError('Installed weight size differs from its manifest')
    fields = inspect_gguf(weight.path)
    pins = (root,) if root == selected else (root, selected)
    model = NativeModel(model_id, source_digest, weight, pins, fields['tokenizer.chat_template'],
        fields['template_sha256'], fields['special_tokens']['<think>'], fields['special_tokens']['</think>'],
        fields['tokenizer.ggml.eos_token_id'], fields['token_count'], fields['qwen35.context_length'])
    model.check()
    return model


@dataclass(frozen=True)
class NativeBackendSpec:
    executable: VerifiedFile
    libraries: tuple[VerifiedFile, ...]
    model: NativeModel
    port: int = 11435
    context_length: int = 8192
    max_tokens: int = 1024
    thinking_tokens: int = 768
    seed: int = 0
    cold_memory_minimum_bytes: int = 7624568755

    def __post_init__(self):
        if (self.executable.path.name != 'llama-server.exe'
                or not self.libraries or len({p.path for p in self.libraries}) != len(self.libraries)
                or any(p.path.parent != self.executable.path.parent for p in self.libraries)
                or type(self.port) is not int or not 1024 <= self.port <= 65535 or self.port == 11434
                or type(self.context_length) is not int or not 0 < self.context_length <= self.model.training_context
                or type(self.max_tokens) is not int or not 0 < self.max_tokens <= MAX_NATIVE_OUTPUT_TOKENS
                or type(self.thinking_tokens) is not int or not 0 < self.thinking_tokens < self.max_tokens
                or self.max_tokens >= self.context_length or type(self.seed) is not int or self.seed != 0
                or type(self.cold_memory_minimum_bytes) is not int
                or self.cold_memory_minimum_bytes < self.model.weight.signature[0] + 1024**3):
            raise ValueError('Invalid pinned native diagnostic specification')
        names = {p.path.name for p in self.libraries}
        if not {'libllama-server-impl.dll', 'libllama-common.dll', 'libllama.dll', 'ggml-base.dll'} <= names:
            raise ValueError('Native backend libraries are not completely pinned')

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.port}'

    def check(self):
        self.executable.check()
        for pin in self.libraries:
            pin.check()
        installed = {p.resolve() for p in self.executable.path.parent.glob('*.dll')}
        if installed != {p.path for p in self.libraries}:
            raise ValueError('Installed native library set changed')
        self.model.check()

    def record(self):
        return dict(executable=self.executable.record(), libraries=[p.record() for p in self.libraries],
            model=self.model.record(), port=self.port, context_length=self.context_length,
            max_tokens=self.max_tokens, thinking_tokens=self.thinking_tokens, seed=self.seed,
            cold_memory_minimum_bytes=self.cold_memory_minimum_bytes)

    def argv(self):
        return native_argv(self.record())


def native_argv(record):
    """One explicit native launch specification, also used in historical replay."""
    return [record['executable']['path'], '--model', record['model']['weight']['path'], '--alias', record['model']['model_id'],
            '--host', '127.0.0.1', '--port', str(record['port']), '--ctx-size', str(record['context_length']),
            '--parallel', '1', '--device', 'none', '--n-gpu-layers', '0', '--threads', '4', '--fit', 'off', '--offline',
            '--jinja', '--reasoning', 'on', '--reasoning-format', 'deepseek',
            '--reasoning-budget', str(record['thinking_tokens']), '--no-warmup', '--no-webui',
            '--no-context-shift', '--cache-ram', '0', '--no-webui-mcp-proxy']


def native_spec_from_record(record):
    """Rehash frozen local assets; never trust serialized stat signatures."""
    model_record = record['model']
    source = model_record['source']
    manifest = Path(source[0]['path'])
    roots = [p for p in manifest.parents if p.name == 'manifests']
    if len(roots) != 1:
        raise ValueError('Ambiguous installed model root')
    model = installed_ollama_model(roots[0].parent, manifest, model_id=model_record['model_id'],
        source_digest=model_record['source_digest'])
    spec = NativeBackendSpec(
        verify_file(record['executable']['path'], record['executable']['sha256']),
        tuple(verify_file(p['path'], p['sha256']) for p in record['libraries']), model,
        **{key:record[key] for key in ('port','context_length','max_tokens','thinking_tokens','seed','cold_memory_minimum_bytes')})
    if json.dumps(spec.record(), sort_keys=True) != json.dumps(record, sort_keys=True):
        raise ValueError('Frozen native asset/configuration record changed')
    spec.check()
    return spec


class OwnedLlamaServer:
    """Explicit process lifetime; neither provider construction nor chat starts it."""
    def __init__(self, spec, *, log_path):
        if not isinstance(spec, NativeBackendSpec):
            raise ValueError('An installed, pinned native specification is required')
        self.spec, self.log_path = spec, Path(log_path)
        self._handle = self._log = self._created = None
        self._job = None
        self.started = False
        self.closed = False
        self.memory_admission = None
        self.cleanup = None

    def start(self):
        import psutil
        if self.started or self.closed:
            raise ValueError('An owned native process is single-use')
        remaining_time()
        self.spec.check()
        self.memory_admission = local_memory_admission(dict(model=self.spec.model.model_id,
            digest=self.spec.model.source_digest, limits=dict(context_length=self.spec.context_length,
                cold_memory_minimum_bytes=self.spec.cold_memory_minimum_bytes, resident_memory_minimum_bytes=1024**3)),
            dict(models=[]), psutil.virtual_memory().available, separate_process=True)
        if not self.memory_admission['admitted']:
            raise MemoryError('Insufficient cold-model memory for the separate native process')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if os.name == 'nt':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', self.spec.port))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open('xb')
        allowed = {'PATH','SYSTEMROOT','WINDIR','SYSTEMDRIVE','COMSPEC','TEMP','TMP','USERPROFILE',
                   'LOCALAPPDATA','APPDATA','PROCESSOR_ARCHITECTURE','NUMBER_OF_PROCESSORS'}
        env = {k:v for k,v in os.environ.items() if k.upper() in allowed}
        env['HF_HUB_OFFLINE'] = '1'
        try:
            if os.name == 'nt':
                from psychology_base.llm.owned_process import WindowsProcessJob
                self._job = WindowsProcessJob()
            self._handle = subprocess.Popen(self.spec.argv(), stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=subprocess.STDOUT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if self._job is not None:
                self._job.assign(self._handle)
            self._created = psutil.Process(self._handle.pid).create_time()
            self.started = True
        except BaseException:
            self.close()
            raise
        return self

    def assert_owned(self, *, listener=True):
        import psutil
        remaining_time()
        self.spec.check()
        if not self.started or self.closed or self._handle.poll() is not None:
            raise ValueError('Owned native backend is not running')
        process = psutil.Process(self._handle.pid)
        if (process.create_time() != self._created or Path(process.exe()).resolve() != self.spec.executable.path
                or process.cmdline() != self.spec.argv()):
            raise ValueError('Native backend process identity changed')
        if listener and not any(c.status == psutil.CONN_LISTEN and c.laddr.ip == '127.0.0.1'
                and c.laddr.port == self.spec.port for c in process.net_connections(kind='tcp')):
            raise ValueError('Owned backend does not own the declared listening socket')
        return dict(pid=self._handle.pid, created=self._created, spec_sha256=hashlib.sha256(
            json.dumps(self.spec.record(), sort_keys=True, separators=(',', ':')).encode()).hexdigest())

    def close(self):
        """Cleanup is independent of an expired request deadline."""
        import psutil
        if self.closed:
            return
        self.closed = True
        targets = []
        try:
            if self._job is not None:
                self.cleanup = self._job.stop()
                # If assignment failed, the job may be empty while our direct
                # Popen child is still alive. Clean that handle separately.
            if self._handle is not None and self._handle.poll() is None:
                parent = psutil.Process(self._handle.pid)
                # The Popen handle remains authoritative on Windows; child PID
                # checks prevent a reused PID from receiving cleanup signals.
                created = parent.create_time()
                if self._created is not None and created != self._created:
                    raise RuntimeError('Refusing cleanup of a changed native process identity')
                targets = [(p, p.create_time()) for p in parent.children(recursive=True)] + [(parent, created)]
                for proc, birth in targets:
                    try:
                        if proc.create_time() != birth:
                            raise RuntimeError('Refusing cleanup of a reused descendant PID')
                        proc.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs([p for p,_ in targets], timeout=5)
                for proc in alive:
                    birth = next(t for p,t in targets if p.pid == proc.pid)
                    if proc.create_time() != birth:
                        raise RuntimeError('Refusing forced cleanup of a reused PID')
                    proc.kill()
                _, alive = psutil.wait_procs(alive, timeout=5)
                if alive:
                    raise RuntimeError('Owned native process cleanup did not finish')
                self._handle.wait(timeout=5)
            if self.cleanup is None:
                self.cleanup = dict(verified_exit=True, tracked_processes=len(targets))
        finally:
            if self._log is not None:
                self._log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
