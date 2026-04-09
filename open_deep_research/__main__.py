import os
import sys

# Enable UTF-8 mode for the entire process (PEP 540).
# This ensures all I/O uses UTF-8 regardless of the system locale,
# preventing encoding errors from LLM/web content on Windows.
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from open_deep_research.cli.app import main

main()
