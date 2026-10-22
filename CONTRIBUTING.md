# Contributing to `jointfm-client`

Thanks for your interest in improving the JointFM Python SDK. This repository
ships the public `jointfm-client` package on PyPI, so every change is shipped to
external users — please read this guide before opening a pull request.

## Ground Rules

- The SDK is licensed under **Apache-2.0**. By contributing, you agree that
  your contribution is licensed under the same terms (inbound = outbound).
- Follow the project rules in [AGENTS.md](AGENTS.md). They take precedence
  over anything stated here.
- Be respectful in issues, reviews, and discussions.

## Reporting Issues

- **Bugs and feature requests:** open a GitHub issue at
  [datarobot/joint-client-python](https://github.com/datarobot/joint-client-python/issues)
  with a minimal reproduction, the SDK version (`python -c "import
  jointfm_client; print(jointfm_client.__version__)"`), and the Python version.
- **Security issues:** do not open a public issue. Follow the disclosure
  process in [SECURITY.md](SECURITY.md).

## Local Setup

Install `uv` and `task` with the same curl-based flow on Ubuntu, macOS, and
AWS Linux, and make sure `~/.local/bin` is on your `PATH` in new shells:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
task setup
```

`task setup` creates `.venv`, installs dev dependencies, registers pre-commit
hooks (including the `commit-msg` hook), and ensures `typos` is available.

Supported development platforms: Ubuntu, AWS Linux, and macOS. Windows is not
supported for development; do not add Windows-specific code paths.

## Development Commands

- `task lint`: run Ruff lint checks (read-only)
- `task format`: run Ruff formatter (rewrites files in place)
- `task license-check`: verify every Python source file has the required copyright/SPDX header
- `task typecheck`: run `ty` static type checks
- `task test`: run unit tests
- `task max-utilization`: send the largest advertised forecast to the configured deployment and print a PASS/FAIL verdict with wall-clock times
- `task coverage`: run tests with coverage enforcement above 90%
- `task build`: build the source distribution and wheel, then validate artifact metadata and contents
- `task check`: run the static code quality gate (typos, lint, format check, type checks)
- `task release:dry`: preview the next SemVer bump without changing any files
- `task release`: cut a SemVer release with Commitizen (writes `CHANGELOG.md`, bumps versions, creates tag)
- `task release:publish`: push the release commit and its tag to `origin`, which triggers the PyPI publish workflow
- `task pre-commit`: run every configured pre-commit hook
- `task leak-guard`: fail when a file in the working tree carries private material in its path or content
- `task leak-guard:history`: fail when a commit carries private material in its message, paths, or files; pass revisions to narrow the scan (`task leak-guard:history -- origin/main..HEAD`)
- `task secret-scan`: fail when gitleaks finds a credential in any commit of the git history
- `task secret-scan:staged`: fail when gitleaks finds a credential in the staged changes
- `task vuln-scan`: fail when a package pinned in `uv.lock` has a known vulnerability

Contributors do not need to add copyright or license headers manually. The `insert-license` pre-commit hook runs [skywalking-eyes](https://github.com/apache/skywalking-eyes) (via the `apache/skywalking-eyes` Docker image, so a running Docker daemon is required) to stamp the standard Apache-2.0 header (`Copyright 2026 DataRobot, Inc. and its affiliates.` followed by the standard "Licensed under the Apache License, Version 2.0" notice) into every `.py` file, and the companion `insert-license-notebooks` hook stamps the same notice into a leading markdown cell of every notebook the first time you run `task pre-commit`. Verify the headers are present at any time with `task license-check`.

Your user must belong to the `docker` group so the hook can reach the daemon:

```shell
sudo usermod -aG docker "$USER"
```

Group membership is captured when a process starts, so it must be in effect **before** the editor or its language server launches — start a fresh login session (or reboot) after adding yourself to the group, otherwise the commit hook inherits the old groups and fails with a Docker permission error.

You must run `task pre-commit` and fix every reported issue before reporting a
change as done. After it succeeds, review the resulting diff and explain why
each hunk is necessary in the pull request.

### Leak guard

This repository is public, so `scripts/leak_guard.py` fails on material that must stay private: proprietary file headers, DataRobot object IDs, internal hosts and private addresses, links to non-public repositories, planning labels, and credential shapes. It runs as a pre-commit hook over the working tree, as a `commit-msg` hook over each message, and in CI over every commit a pull request or push adds. Commits published before the guard existed cannot be cleaned, so CI does not scan the full history.

Some rules cannot be published because they name what they guard. Maintainers keep those in a git-ignored `.leak-patterns.private` at the repository root: one Python regular expression per line, blank lines and `#` comment lines skipped. The guard reports whether it loaded the file and runs the public rules alone when it is absent, as in CI. A dangling link fails. A passing guard does not prove a change safe to publish.

### Secret and vulnerability scans

The leak guard knows DataRobot-specific material; [gitleaks](https://github.com/gitleaks/gitleaks) covers generic credential shapes. It runs as a pre-commit hook over the staged changes, and the `Security` workflow scans the full history on every pull request, every push to `main`, and once a week. The same workflow runs [osv-scanner](https://github.com/google/osv-scanner) over `uv.lock`. Both scanners run from Docker images pinned by digest in `Taskfile.yaml`.

`pyproject.toml` keeps new releases out of `uv.lock` for a cooldown period (`exclude-newer`). When an advisory's fix is still inside that window, add it to [`osv-scanner.toml`](osv-scanner.toml) with an `ignoreUntil` date and a reason, then bump the lock once the fix clears the cooldown. The waiver lapses on its own, so a forgotten bump fails the scan again.

## Branch And Pull Request Flow

1. Fork the repository (or branch directly if you are a DataRobot maintainer).
2. Create a feature branch off `main` with a descriptive name.
3. Make focused commits that pass `task pre-commit` locally.
4. Open a pull request against `main`. Describe what changed, why, and how
   you verified it.
5. Keep pull requests small and reviewable. Refactors, formatting-only
   changes, and behavior changes should be separate commits and ideally
   separate pull requests.

## Versioning & Commits

The package follows strict [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are cut with [Commitizen](https://commitizen-tools.github.io/commitizen/), driven by [Conventional Commits](https://www.conventionalcommits.org/), so the commit log is the source of truth for what a release contains.

### Commit message format

Squash merges use GitHub's default message, usually the pull request title, as the commit subject on `main`. For `cz bump` to see a change, that subject must follow:

```text
<type>(<optional scope>): <imperative summary>
```

Examples:

```text
feat(adapters): add quantile forecast helper
fix(transport): retry on idempotent 5xx responses
perf(adapters): cache schema validation across forecast batches
refactor(configuration): split URL resolution into helpers
docs(readme): document the release workflow
test(transport): cover retry on 502/503/504
build(deps): bump pydantic to 2.13.4
ci(pre-commit): pin commitizen to v3.31
chore(repo): move fixtures to tests/fixtures/v1
```

#### Commit types

The full set of accepted types comes from the `cz_conventional_commits` rule set referenced above. Pick the type that matches the **primary intent** of the commit; split commits that mix concerns. Only `feat`, `fix`, `refactor`, and `perf` (plus breaking-change markers) drive a SemVer bump — every other type is recorded in `git log` but does not, on its own, cause `cz bump` to cut a new version.

| Type | When to use | SemVer bump |
| --- | --- | --- |
| `feat` | A user-visible feature: new public API, new CLI flag, new helper exposed to callers. | **minor** |
| `fix` | A bug fix in shipped behavior — the symptom is observable to callers or operators. | **patch** |
| `perf` | A change that improves performance without changing observable behavior. | **patch** |
| `refactor` | An internal restructure that neither adds a feature nor fixes a bug (renames, moves, extractions, internal type changes). | **patch** |
| `docs` | Documentation-only changes (READMEs, prose, docstrings, comments, example payloads, API reference). | none |
| `test` | Adding, fixing, or restructuring tests, fixtures, or test helpers with no production-code change. | none |
| `build` | Build system, packaging, or dependency-pinning changes (`pyproject.toml`, lockfile, wheel build hooks). | none |
| `ci` | CI configuration changes (GitHub Actions workflows, `.pre-commit-config.yaml`, release hooks). | none |
| `chore` | Repository maintenance not covered above (tooling tweaks, repo-level renames, housekeeping, fixture moves). | none |
| `style` | Pure formatting or whitespace, no behavior change — rare here because Ruff format runs in pre-commit. | none |
| `revert` | Reverts a previous commit; the body should include `Refs: <sha>`. Re-add `!` or a `BREAKING CHANGE:` footer if the reverted commit was a breaking change. | none |
| any type with `!` or a `BREAKING CHANGE:` footer | A breaking change to public API, configuration schema, environment variables, or the service wire contract. | **major**¹ |

¹ This project sets `major_version_zero = true` in `pyproject.toml`, so breaking changes are downgraded to a **minor** bump while the SDK is on `0.x`. They will become major bumps once the SDK ships `1.0.0`.

A release window that contains only `none`-bump commits is not releasable on its own: `cz bump` exits without writing a new version. Either land a `feat`/`fix`/`refactor`/`perf` first, or force the bump explicitly with `task release -- --increment PATCH`.

#### Breaking changes

Declare a breaking change with `!` after the type, e.g. `feat(adapters)!: rename forecast() to predict()`, or — preferred when the change needs explanation — with an explicit footer:

```text
feat(adapters): rename forecast() to predict()

BREAKING CHANGE: forecast() is gone; callers must use predict().
```

Commits on a feature branch are discarded by the squash merge, so their messages are not checked against this format.

## Public API Surface

The SDK's public surface is everything re-exported from
[`src/jointfm_client/__init__.py`](src/jointfm_client/__init__.py). Names that
start with an underscore are implementation details.

- Adding to `__all__` is a `feat`.
- Removing from `__all__`, renaming an exported name, or changing the shape of
  a public class, function, or exception is a **breaking change**. Use the
  Conventional Commits `!` marker or a `BREAKING CHANGE:` footer.
- The wire contract pinned by `schema_version` is owned by the JointFM
  service team. Do not change the request or response shape unilaterally.
- New public exports must also be documented in
  [docs/api-reference.md](docs/api-reference.md) in the same change.

## Tests

- Add or update tests for every behavior change. The pre-commit gate enforces a
  coverage floor; uncovered new code will fail the gate.
- Tests must be fast. The live DataRobot smoke test
  ([tests/test_live_smoke.py](tests/test_live_smoke.py)) is gated by environment
  variables and skipped by default.

## Releases

Releases are cut by maintainers using Commitizen.

### First-time tag seed (one-off)

Commitizen bumps **from** the tag matching the current `version =` in `pyproject.toml`. A brand-new repo has no such tag, so the first `task release` or `task release:dry` will fail with a clear message. Seed it once:

```bash
git tag -a v0.0.1 -m 'Seed initial release tag for Commitizen'
git push --tags
```

After that, every future `task release` finds its base tag automatically.

### Cutting a release

```bash
task release:dry      # preview the next version + CHANGELOG entries
task release          # bump, write CHANGELOG.md, create the annotated tag
task release:publish  # push the bump commit and the tag
```

`task release` first runs `task release:check` (clean tree, on `main`, in sync with `origin/main`), then calls `cz bump` which:

- reads commits since the last `v*` tag,
- picks the SemVer bump from the types it sees,
- updates `CHANGELOG.md`,
- bumps `version =` in `pyproject.toml` and the "Current SDK package version" line in [README.md](README.md),
- commits the bump and creates the annotated tag.

Publishing is a separate task on purpose, so you can inspect the inferred bump before anything leaves your machine — Commitizen derives the version from commit messages, and a stray `feat:` where you meant `fix:` is only fixable while the release is still local. Override the inferred bump level when needed: `task release -- --increment minor`.

`task release:publish` re-checks that the working tree is clean, that you are on `main`, that the tag points at `HEAD`, and that `origin` does not already have the tag, prints the commits about to be pushed, asks for confirmation on a terminal, and then pushes `main` and that single tag. It pushes one explicit tag rather than `git push --tags`, so unrelated local tags are never published.

Pushing the `v*` tag triggers the [`Publish to PyPI`](.github/workflows/publish.yml) workflow, which rebuilds and validates the distribution with `task build` and uploads it to PyPI via [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish) — keeping the git tag, the wheel filename, `jointfm_client.__version__`, and the PyPI version all in lockstep.

The workflow authenticates with PyPI through [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no API-token secret is stored in the repository. Before the first release, configure it once on PyPI:

- register a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for the `jointfm-client` project pointing at owner `datarobot`, repository `joint-client-python`, workflow `publish.yml`, and environment `pypi`;
- create a GitHub Actions environment named `pypi` on the repository (optionally gated with required reviewers) so the publish job can run.

After pulling these changes for the first time, run `uv run pre-commit install` (or `task setup`) once so the new `commit-msg` hook is registered with git.

### Hand-editing the version (don't)

The `version =` line in [`pyproject.toml`](pyproject.toml) and the "Current SDK package version" line in [README.md](README.md) are **owned by `cz bump`** — treat them the way you'd treat a lockfile. Each release rewrites both of them atomically.

Nothing in the package source carries a version literal: `jointfm_client.contract.PACKAGE_VERSION` reads the installed distribution metadata that the build backend generates from `[project].version`, and `jointfm_client.__version__` is that same value. They follow a bump automatically, so there is nothing to hand-edit and nothing that can drift.

If you hand-edit them, the next `task release` will catch you:

- if you bumped only some lines, `cz bump` aborts with "Configured files cannot be updated, check consistency";
- if you bumped them all but never created the matching `vX.Y.Z` tag, `task release:require-tag` refuses to run and tells you to seed the tag.

If you genuinely need to set a specific version outside the normal release flow (bootstrap, recovery from a botched state), use the escape hatch:

```bash
uv run cz bump --files-only X.Y.Z
```

That atomically rewrites every `version_files` line to `X.Y.Z` without committing or tagging. After it returns, commit and tag manually.

## Git Safety

Never run destructive git operations unless a maintainer explicitly asks for
that specific command — no `git push --force`, no `git reset --hard`, no
history rewrites, no `--no-verify` to bypass hooks, and no discarding of
uncommitted work.
