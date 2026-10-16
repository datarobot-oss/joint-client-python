# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail when material that must stay private is about to be published.

The guard matches file paths, file contents, and commit messages line by line
against two rule sets. The public rules below are safe to publish. The private
rules name the very material they guard, so maintainers keep them outside
version control in the git-ignored ``PRIVATE_RULES_FILE``; the guard reports
whether it loaded them and runs the public rules alone when the file is absent.

Modes:

- ``tree``: every tracked file and every untracked file that is not ignored, as
  it is in the working tree.
- ``message <file>``: one commit message, as the ``commit-msg`` hook passes it.
- ``history [<revision> ...]``: every commit message, path, and file version
  reachable from any ref, because a published history cannot be cleaned
  afterwards. Revisions such as ``origin/main..HEAD`` narrow the scan to the
  commits they select and the file versions those commits introduce, for a
  repository whose history was published before the guard existed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PRIVATE_RULES_FILE = ".leak-patterns.private"
BINARY_SNIFF_BYTES = 8192
EVERY_REF = ("--all",)
PUBLIC_DATAROBOT_REPOSITORIES = ("joint-client-python", "joint-inference-python")
PUBLIC_DATAROBOT_HOSTS = ("app", "app.eu", "app.jp", "docs", "www")


@dataclass(frozen=True)
class Rule:
    """One named pattern that marks a line as private material."""

    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    """One rule match: where it is, which rule matched, and the matched text."""

    location: str
    line: int | None
    rule: str
    excerpt: str

    def render(self) -> str:
        """Return the finding as one report line."""
        position = (
            self.location if self.line is None else f"{self.location}:{self.line}"
        )
        return f"{position}: [{self.rule}] {self.excerpt}"


def _alternatives(names: Iterable[str]) -> str:
    return "|".join(re.escape(name) for name in names)


# Every pattern is written so that its own source text does not match it, which
# lets the guard scan this file without an exemption.
PUBLIC_RULES = (
    Rule(
        "proprietary-header",
        re.compile(
            r"(?i)\bproprietary\s+and\s+confidential\b|LicenseRef-\s*Proprietary"
        ),
    ),
    Rule("datarobot-object-id", re.compile(r"\b[0-9a-f]{24}\b")),
    Rule(
        "internal-host",
        re.compile(
            r"(?i)\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|corp|intranet|lan)\b"
            r"|\bip(?:-\d{1,3}){4}\b"
            r"|\b[a-z0-9.-]+\.amazonaws\.com\b"
        ),
    ),
    Rule(
        "private-ip-address",
        re.compile(
            r"\b(?:10(?:\.\d{1,3}){3}"
            r"|192\.168(?:\.\d{1,3}){2}"
            r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
        ),
    ),
    Rule(
        "datarobot-host",
        re.compile(
            rf"(?i)\b(?!(?:{_alternatives(PUBLIC_DATAROBOT_HOSTS)})\.datarobot\.com\b)"
            r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.datarobot\.com\b"
        ),
    ),
    Rule(
        "private-repository-link",
        re.compile(
            rf"(?i)github\.com/datarobot/(?!(?:{_alternatives(PUBLIC_DATAROBOT_REPOSITORIES)})(?![\w.-]))[\w.-]+"
        ),
    ),
    Rule(
        "planning-label",
        re.compile(r"(?i)\b(?:phase|milestone|sprint)[\s_-]*\d+[a-z0-9]*\b"),
    ),
    Rule(
        "credential",
        re.compile(
            r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"
            r"|\bgithub_pat_\w{22,}"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bhf_[A-Za-z0-9]{30,}\b"
            r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        ),
    ),
)
PRIVATE_RULES_FILE_RULE = Rule(
    "private-rules-file-published",
    re.compile(rf"(?:^|/){re.escape(PRIVATE_RULES_FILE)}$"),
)


def load_private_rules(path: Path) -> tuple[Rule, ...]:
    """Read the private rules: one Python regular expression per line.

    Blank lines and lines starting with ``#`` are skipped. Each rule is named
    after its line number, so a report never repeats the private pattern itself.

    Raises:
        OSError: when ``path`` cannot be read, including a dangling symbolic link.
        re.error: when a line is not a valid regular expression.
        ValueError: when the file holds no pattern.
    """
    rules: list[Rule] = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        rules.append(Rule(f"private:{number}", re.compile(text)))
    if not rules:
        raise ValueError(f"{path} holds no patterns; delete it or add patterns")
    return tuple(rules)


def resolve_rules(root: Path) -> tuple[tuple[Rule, ...], str]:
    """Return the content rules for ``root`` and a line saying which sets are active.

    A dangling ``PRIVATE_RULES_FILE`` link is not treated as absent: loading it
    raises, so a moved private list never silently drops the private rules.
    """
    path = root / PRIVATE_RULES_FILE
    if not path.is_symlink() and not path.exists():
        return (
            PUBLIC_RULES,
            f"leak guard: public rules only; {PRIVATE_RULES_FILE} is absent",
        )
    private = load_private_rules(path)
    status = f"leak guard: public rules and {len(private)} private rules from {PRIVATE_RULES_FILE}"
    return PUBLIC_RULES + private, status


def scan_text(location: str, text: str, rules: Sequence[Rule]) -> list[Finding]:
    """Match every line of ``text`` against every rule."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in rules:
            match = rule.pattern.search(line)
            if match is not None:
                findings.append(Finding(location, number, rule.name, match.group(0)))
    return findings


def scan_path(path: str, rules: Sequence[Rule]) -> list[Finding]:
    """Match a repository-relative path against the rules and the private-file rule."""
    findings: list[Finding] = []
    for rule in (*rules, PRIVATE_RULES_FILE_RULE):
        match = rule.pattern.search(path)
        if match is not None:
            findings.append(Finding(path, None, rule.name, match.group(0)))
    return findings


def _decode_text(data: bytes) -> str | None:
    """Return ``data`` as text, or ``None`` for binary content."""
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return None
    return data.decode("utf-8", errors="replace")


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], input=stdin, capture_output=True, check=True
    )
    return completed.stdout


def scan_tree(root: Path, rules: Sequence[Rule]) -> list[Finding]:
    """Scan every tracked and every untracked, non-ignored file in the working tree.

    A symbolic link is scanned by its target path, which is what git stores. A
    tracked file deleted from the working tree contributes only its path.
    """
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = sorted({path for path in listing.decode("utf-8").split("\0") if path})
    findings: list[Finding] = []
    for relative in paths:
        findings.extend(scan_path(relative, rules))
        full = root / relative
        if full.is_symlink():
            text: str | None = os.readlink(full)
        elif full.exists():
            text = _decode_text(full.read_bytes())
        else:
            continue
        if text is not None:
            findings.extend(scan_text(relative, text, rules))
    return findings


def scan_message(path: Path, rules: Sequence[Rule]) -> list[Finding]:
    """Scan a commit message file, skipping the ``#`` comment lines git strips."""
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = "\n".join("" if line.startswith("#") else line for line in lines)
    return scan_text("commit message", kept, rules)


def _reachable_blobs(root: Path, revisions: Sequence[str]) -> dict[str, str]:
    """Map every blob ``revisions`` reach to the first path it was seen at."""
    blobs: dict[str, str] = {}
    listing = _git(root, "rev-list", "--objects", *revisions).decode("utf-8")
    entries = [line.split(" ", 1) for line in listing.splitlines()]
    named = [(entry[0], entry[1]) for entry in entries if len(entry) == 2]
    if not named:
        return blobs
    request = "".join(f"{sha}\n" for sha, _ in named).encode("ascii")
    kinds = _git(
        root, "cat-file", "--batch-check=%(objectname) %(objecttype)", stdin=request
    )
    for (sha, path), line in zip(
        named, kinds.decode("ascii").splitlines(), strict=True
    ):
        if line.split(" ")[1] == "blob":
            blobs.setdefault(sha, path)
    return blobs


def _read_blobs(root: Path, shas: Sequence[str]) -> dict[str, bytes]:
    """Read many blobs through one ``git cat-file --batch`` call."""
    output = _git(
        root,
        "cat-file",
        "--batch",
        stdin="".join(f"{sha}\n" for sha in shas).encode("ascii"),
    )
    contents: dict[str, bytes] = {}
    offset = 0
    for sha in shas:
        header_end = output.index(b"\n", offset)
        size = int(output[offset:header_end].split(b" ")[2])
        start = header_end + 1
        contents[sha] = output[start : start + size]
        offset = start + size + 1
    return contents


def scan_history(
    root: Path, rules: Sequence[Rule], revisions: Sequence[str] = EVERY_REF
) -> list[Finding]:
    """Scan every commit message, path, and blob that ``revisions`` select.

    ``revisions`` are passed to ``git log`` and ``git rev-list --objects``; the
    default selects everything reachable from any ref. A range such as
    ``base..HEAD`` selects the commits after ``base`` and only the blobs they
    reach that ``base`` does not. An unknown revision fails in git.
    """
    findings: list[Finding] = []
    log = _git(root, "log", *revisions, "--format=%H%x00%B%x00").decode(
        "utf-8", errors="replace"
    )
    fields = log.split("\0")
    for sha, message in zip(fields[0::2], fields[1::2], strict=False):
        commit = sha.strip()
        if commit:
            findings.extend(scan_text(f"commit {commit[:12]} message", message, rules))
    blobs = _reachable_blobs(root, revisions)
    for path in sorted(set(blobs.values())):
        findings.extend(scan_path(path, rules))
    contents = _read_blobs(root, sorted(blobs))
    for sha, path in sorted(blobs.items(), key=lambda item: item[1]):
        text = _decode_text(contents[sha])
        if text is not None:
            findings.extend(scan_text(f"{path} (blob {sha[:12]})", text, rules))
    return findings


def repository_root() -> Path:
    """Return the top level of the git repository the guard runs in."""
    return Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel").decode("utf-8").strip()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard and return 1 when any rule matched."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser(
        "tree", help="scan the working tree's tracked and untracked, non-ignored files"
    )
    history = modes.add_parser(
        "history", help="scan everything reachable from any ref, or the given revisions"
    )
    history.add_argument(
        "revisions",
        nargs="*",
        help="git revisions or ranges to scan instead of every ref, e.g. origin/main..HEAD",
    )
    message = modes.add_parser("message", help="scan one commit message file")
    message.add_argument("message_file", type=Path)
    args = parser.parse_args(argv)

    root = repository_root()
    rules, status = resolve_rules(root)
    print(status)
    if args.mode == "tree":
        findings = scan_tree(root, rules)
    elif args.mode == "history":
        findings = scan_history(root, rules, args.revisions or EVERY_REF)
    else:
        findings = scan_message(args.message_file, rules)
    for finding in findings:
        print(finding.render())
    if findings:
        print(
            f"leak guard: {len(findings)} finding(s); remove the material before it is committed or pushed"
        )
        return 1
    print("leak guard: no findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
