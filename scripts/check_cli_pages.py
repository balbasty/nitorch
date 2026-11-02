#!/usr/bin/env python3
"""Check that every registered `nitorch` command has a reference page.

Commands register themselves with `commands['name'] = fn` somewhere under
`nitorch/cli`. This script finds those registrations with a regular
expression (so that it does not need to import torch) and checks that each
command has a page under `docs/cli/reference/` that is listed in the
navigation of `zensical.toml`.

Run from the repository root:

    python scripts/check_cli_pages.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "nitorch" / "cli"
REFERENCE = ROOT / "docs" / "cli" / "reference"
CONFIG = ROOT / "zensical.toml"

# Registered names that are deliberately left undocumented.
HIDDEN = {
    "register2",  # backward-compatible alias of `register`
}

pattern = re.compile(r"""commands\[\s*['"]([^'"]+)['"]\s*\]\s*=""")
commands = set()
for path in CLI.rglob("*.py"):
    if path.name == "cli.py" and path.parent == CLI:
        continue
    commands.update(pattern.findall(path.read_text()))
commands -= HIDDEN

pages = {p.stem: p for p in REFERENCE.rglob("*.md") if p.stem != "index"}
config = CONFIG.read_text()

errors = []
for name in sorted(commands):
    if name not in pages:
        errors.append(f"command `{name}` has no page in {REFERENCE}")
        continue
    rel = pages[name].relative_to(ROOT / "docs").as_posix()
    if f'"{rel}"' not in config:
        errors.append(f"page {rel} is not listed in zensical.toml")
for name in sorted(set(pages) - commands):
    errors.append(f"page {pages[name]} documents an unknown command")

if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"All {len(commands)} commands are documented.")
