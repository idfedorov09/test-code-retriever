import hashlib
import logging
import re

_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{120,}={0,2})(?![A-Za-z0-9+/=])"
)


class Base64TruncatingFilter(logging.Filter):
    def __init__(self, max_show: int = 24):
        super().__init__()
        self.max_show = max_show

    def _short(self, s: str) -> str:
        orig_len = len(s)
        head = s[: self.max_show // 2]
        tail = s[-(self.max_show - len(head)) :]
        digest = hashlib.sha256(s.encode("ascii", "ignore")).hexdigest()[:8]
        return f"{head}…{tail} <base64 len={orig_len} sha256={digest}>"

    def _sanitize_str(self, text: str) -> str:
        return _BASE64_RE.sub(lambda m: self._short(m.group(1)), text)

    def _sanitize_any(self, v):
        if isinstance(v, str):
            return self._sanitize_str(v)
        if isinstance(v, dict):
            return {k: self._sanitize_any(val) for k, val in v.items()}
        if isinstance(v, (list, tuple, set)):
            t = type(v)
            return t(self._sanitize_any(x) for x in v)
        return v

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = self._sanitize_str(record.msg)

            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(self._sanitize_any(a) for a in record.args)
                else:
                    record.args = self._sanitize_any(record.args)

            skip = {
                "name",
                "msg",
                "args",
                "levelno",
                "levelname",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }
            for k, v in list(record.__dict__.items()):
                if k in skip:
                    continue
                record.__dict__[k] = self._sanitize_any(v)
        except Exception:
            pass
        return True
