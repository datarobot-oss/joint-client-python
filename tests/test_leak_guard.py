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

"""Tests for the leak guard that keeps private material out of this repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import leak_guard

REPO_ROOT = Path(__file__).resolve().parents[1]

# Samples are assembled at runtime so this file's own source never matches a rule.
OBJECT_ID = "6a0" + "23f4f844e91557a9fc1f3"
HEADER = "proprietary and " + "confidential to the owner"
SAMPLES = {
    "proprietary-header": HEADER,
    "datarobot-object-id": f"deployment {OBJECT_ID}",
    "internal-host": "gpu-7.cluster" + ".internal",
    "private-ip-address": "10.2" + ".3.4",
    "datarobot-host": "shell" + ".datarobot.com",
    "private-repository-link": "github.com/datarobot/" + "secret-repo",
    "planning-label": "Phase" + " 3g",
    "credential": "AKIA" + "ABCDEFGHIJKLMNOP",
}
PUBLIC_TEXTS = (
    "https://app" + ".datarobot.com/api/v2",
    "https://github.com/datarobot/" + "joint-client-python",
    "contact: someone@" + "datarobot.com",
    "JointFM" + " 1 serves 2 horizons",
    '"generation": "jointfm-2"',
    "sha256 " + "ab" * 32,
    "phase shift of the seasonal component",
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return an empty git repository that ignores the private rules file."""
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".gitignore").write_text(
        f"/{leak_guard.PRIVATE_RULES_FILE}\n", encoding="utf-8"
    )
    return tmp_path


def _rules_hit(findings: list[leak_guard.Finding]) -> set[str]:
    return {finding.rule for finding in findings}


@pytest.mark.parametrize(("rule", "text"), sorted(SAMPLES.items()))
def test_each_public_rule_flags_its_sample(rule: str, text: str) -> None:
    """Each public rule flags its own sample and no other rule does."""
    findings = leak_guard.scan_text("sample", text, leak_guard.PUBLIC_RULES)
    assert _rules_hit(findings) == {rule}


@pytest.mark.parametrize("text", PUBLIC_TEXTS)
def test_public_rules_pass_publishable_text(text: str) -> None:
    """Publishable text, including the client's generation names, passes the public rules."""
    assert leak_guard.scan_text("sample", text, leak_guard.PUBLIC_RULES) == []


def test_this_repository_passes_its_own_guard() -> None:
    """This repository's working tree passes the rules the guard resolves here."""
    rules, _ = leak_guard.resolve_rules(REPO_ROOT)
    assert [
        finding.render() for finding in leak_guard.scan_tree(REPO_ROOT, rules)
    ] == []


def test_private_rules_are_absent_by_default_and_reported(repo: Path) -> None:
    """Without a private rules file the guard runs the public rules and says so."""
    rules, status = leak_guard.resolve_rules(repo)
    assert rules == leak_guard.PUBLIC_RULES
    assert "public rules only" in status


def test_private_rules_load_and_name_rules_by_line(repo: Path) -> None:
    """Private rules load from the file and are named by line, never by pattern."""
    (repo / leak_guard.PRIVATE_RULES_FILE).write_text(
        "# comment\n\n\\bwidget\\b\n", encoding="utf-8"
    )
    rules, status = leak_guard.resolve_rules(repo)
    assert "1 private rules" in status
    findings = leak_guard.scan_text("sample", "a widget here", rules)
    assert [(finding.rule, finding.excerpt) for finding in findings] == [
        ("private:3", "widget")
    ]


def test_dangling_private_rules_link_fails(repo: Path) -> None:
    """A dangling private rules link fails instead of dropping the private rules."""
    (repo / leak_guard.PRIVATE_RULES_FILE).symlink_to(repo / "moved-away.txt")
    with pytest.raises(FileNotFoundError):
        leak_guard.resolve_rules(repo)


def test_private_rules_file_without_patterns_fails(repo: Path) -> None:
    """A private rules file without any pattern fails."""
    (repo / leak_guard.PRIVATE_RULES_FILE).write_text(
        "# only a comment\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="holds no patterns"):
        leak_guard.resolve_rules(repo)


def test_tree_scans_paths_contents_untracked_files_and_link_targets(repo: Path) -> None:
    """The tree scan covers paths, contents, untracked files, and link targets, not binaries."""
    (repo / "tracked.md").write_text(f"line one\n{HEADER}\n", encoding="utf-8")
    _git(repo, "add", "tracked.md")
    notes = "phase" + "2-notes.md"
    (repo / notes).write_text("clean\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\0" + HEADER.encode("utf-8"))
    (repo / "link").symlink_to("/srv/build" + ".internal/data")
    findings = leak_guard.scan_tree(repo, leak_guard.PUBLIC_RULES)
    assert sorted(
        (finding.location, finding.line, finding.rule) for finding in findings
    ) == [
        ("link", 1, "internal-host"),
        (notes, None, "planning-label"),
        ("tracked.md", 2, "proprietary-header"),
    ]


def test_tree_skips_ignored_private_rules_file(repo: Path) -> None:
    """The git-ignored private rules file is not scanned as content."""
    (repo / leak_guard.PRIVATE_RULES_FILE).write_text(
        "\\bwidget\\b\n", encoding="utf-8"
    )
    assert leak_guard.scan_tree(repo, leak_guard.PUBLIC_RULES) == []


def test_tracked_private_rules_file_is_a_finding(repo: Path) -> None:
    """A tracked private rules file is itself a finding."""
    (repo / leak_guard.PRIVATE_RULES_FILE).write_text(
        "\\bwidget\\b\n", encoding="utf-8"
    )
    _git(repo, "add", "--force", leak_guard.PRIVATE_RULES_FILE)
    findings = leak_guard.scan_tree(repo, leak_guard.PUBLIC_RULES)
    assert _rules_hit(findings) == {"private-rules-file-published"}


def test_message_skips_git_comment_lines(tmp_path: Path) -> None:
    """A commit message scan skips the comment lines git strips."""
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text(
        f"feat: Add loader\n\nRefs {OBJECT_ID}\n# {HEADER}\n", encoding="utf-8"
    )
    findings = leak_guard.scan_message(message, leak_guard.PUBLIC_RULES)
    assert [(finding.line, finding.rule) for finding in findings] == [
        (3, "datarobot-object-id")
    ]


def test_history_finds_material_deleted_in_a_later_commit(repo: Path) -> None:
    """The history scan finds material a later commit removed from the tree."""
    (repo / "notes.md").write_text(f"{HEADER}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"docs: Add notes for deployment {OBJECT_ID}")
    (repo / "notes.md").write_text("clean\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "docs: Clean notes")
    assert leak_guard.scan_tree(repo, leak_guard.PUBLIC_RULES) == []
    findings = leak_guard.scan_history(repo, leak_guard.PUBLIC_RULES)
    rendered = sorted(finding.rule for finding in findings)
    assert rendered == ["datarobot-object-id", "proprietary-header"]
    assert any(finding.location.startswith("notes.md (blob ") for finding in findings)


def test_history_of_a_clean_repository_passes(repo: Path) -> None:
    """A clean history passes."""
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: Add readme")
    assert leak_guard.scan_history(repo, leak_guard.PUBLIC_RULES) == []


def test_main_returns_failure_on_findings(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line returns 1 on findings and 0 once they are removed."""
    (repo / "notes.md").write_text(f"{HEADER}\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    assert leak_guard.main(["tree"]) == 1
    output = capsys.readouterr().out
    assert "public rules only" in output
    assert "notes.md:1: [proprietary-header]" in output
    (repo / "notes.md").write_text("clean\n", encoding="utf-8")
    assert leak_guard.main(["tree"]) == 0


def test_history_range_scans_only_the_commits_it_selects(repo: Path) -> None:
    """A revision range scans only the commits it selects and the blobs they add."""
    (repo / "notes.md").write_text(f"{HEADER}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"docs: Add notes for deployment {OBJECT_ID}")
    _git(repo, "tag", "published")
    (repo / "notes.md").write_text("clean\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "docs: Clean notes")
    assert (
        leak_guard.scan_history(repo, leak_guard.PUBLIC_RULES, ["published..HEAD"])
        == []
    )
    (repo / "new.md").write_text("Phase" + " 7\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: Add new notes")
    findings = leak_guard.scan_history(
        repo, leak_guard.PUBLIC_RULES, ["published..HEAD"]
    )
    assert [(finding.rule, finding.excerpt) for finding in findings] == [
        ("planning-label", "Phase" + " 7")
    ]


def test_main_history_accepts_revisions(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The history command narrows its scan to the revisions it is given."""
    (repo / "notes.md").write_text(f"{HEADER}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: Add notes")
    _git(repo, "tag", "published")
    monkeypatch.chdir(repo)
    assert leak_guard.main(["history"]) == 1
    assert leak_guard.main(["history", "published..HEAD"]) == 0
    assert "leak guard: no findings" in capsys.readouterr().out
