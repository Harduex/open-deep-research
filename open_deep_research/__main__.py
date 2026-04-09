import os
import sys

# Enable UTF-8 mode for the entire process (PEP 540).
os.environ["PYTHONUTF8"] = "1"

# Replace stdout/stderr with UTF-8 wrappers.  reconfigure() alone is
# insufficient on Windows because Rich (and other libraries) may
# capture a reference to the pre-reconfigured stream at import time.
# Replacing the stream objects guarantees every subsequent writer —
# including Rich Console — inherits UTF-8 + replace error handling.
import io  # noqa: E402

for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name)
    if hasattr(_stream, "buffer"):
        _new = io.TextIOWrapper(
            _stream.buffer, encoding="utf-8", errors="replace", line_buffering=_stream.line_buffering,
        )
        setattr(sys, _name, _new)

from open_deep_research.cli.app import main

main()
