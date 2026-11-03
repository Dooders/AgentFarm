#!/usr/bin/env python
"""Create the GitHub labels, milestones, project and issues for the 2026-11 engine review.

Reads ``docs/research/reviews/2026-11-engine-review/issues.yaml`` and drives the ``gh`` CLI.
Idempotent: a state file maps issue ids to issue numbers so re-running only creates what is
missing and refreshes cross references.

Usage (from the repository root, with ``gh auth status`` showing write access to the repo):

    python scripts/engine_review/bootstrap_github_project.py --dry-run
    python scripts/engine_review/bootstrap_github_project.py
    python scripts/engine_review/bootstrap_github_project.py --skip-project   # no `project` scope

Projects (v2) need the ``project`` token scope: ``gh auth refresh -s project``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "docs/research/reviews/2026-11-engine-review/issues.yaml"
DEFAULT_STATE = REPO_ROOT / "docs/research/reviews/2026-11-engine-review/.issue_numbers.json"
ID_REF = re.compile(r"#(M\d(?:-[A-Z]\d?)?)\b")


@dataclass
class Gh:
    """Thin wrapper around the gh CLI that honours --dry-run."""

    dry_run: bool
    repo: str
    log: list[str] = field(default_factory=list)

    def run(self, *args: str, capture: bool = True, allow_fail: bool = False) -> str:
        cmd = ["gh", *args]
        printable = " ".join(shlex.quote(a) for a in cmd)
        self.log.append(printable)
        if self.dry_run:
            print(f"[dry-run] {printable}")
            return ""
        result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
        if result.returncode != 0 and not allow_fail:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"command failed: {printable}")
        return (result.stdout or "").strip()

    def api(self, endpoint: str, *fields: str, method: str = "GET", allow_fail: bool = False) -> Any:
        args = ["api", endpoint, "--method", method]
        for f in fields:
            args.extend(["-f", f])
        out = self.run(*args, allow_fail=allow_fail)
        if not out:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    ids = [issue["id"] for issue in spec["issues"]]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(f"duplicate issue ids in spec: {sorted(duplicates)}")
    known = set(ids)
    label_names = {lbl["name"] for lbl in spec["labels"]}
    milestone_titles = {m["title"] for m in spec["milestones"]}
    for issue in spec["issues"]:
        for dep in issue.get("blocked_by", []):
            if dep not in known:
                raise SystemExit(f"{issue['id']}: unknown blocked_by id {dep}")
        if issue.get("epic") is not None and issue["epic"] not in known:
            raise SystemExit(f"{issue['id']}: unknown epic id {issue['epic']}")
        for lbl in issue["labels"]:
            if lbl not in label_names:
                raise SystemExit(f"{issue['id']}: label {lbl!r} not declared in labels:")
        if issue["milestone"] not in milestone_titles:
            raise SystemExit(f"{issue['id']}: milestone {issue['milestone']!r} not declared")
    return spec


def load_state(path: Path) -> dict[str, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict[str, int], dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_labels(gh: Gh, labels: list[dict[str, str]]) -> None:
    for lbl in labels:
        gh.run(
            "label", "create", lbl["name"],
            "--repo", gh.repo,
            "--color", lbl["color"],
            "--description", lbl.get("description", ""),
            "--force",
        )


def ensure_milestones(gh: Gh, milestones: list[dict[str, str]]) -> dict[str, int]:
    existing: dict[str, int] = {}
    if not gh.dry_run:
        data = gh.api(f"repos/{gh.repo}/milestones?state=all&per_page=100") or []
        existing = {m["title"]: m["number"] for m in data}
    numbers: dict[str, int] = {}
    for m in milestones:
        if m["title"] in existing:
            numbers[m["title"]] = existing[m["title"]]
            continue
        created = gh.api(
            f"repos/{gh.repo}/milestones",
            f"title={m['title']}",
            f"description={m.get('description', '').strip()}",
            method="POST",
        )
        numbers[m["title"]] = created["number"] if isinstance(created, dict) else -1
    return numbers


def placeholder_body(issue: dict[str, Any]) -> str:
    header = f"<!-- engine-review-id: {issue['id']} -->\n"
    return header + issue["body"].rstrip() + "\n"


def final_body(issue: dict[str, Any], spec: dict[str, Any], state: dict[str, int]) -> str:
    def ref(issue_id: str) -> str:
        number = state.get(issue_id)
        return f"#{number}" if number else f"`{issue_id}`"

    body = ID_REF.sub(lambda m: ref(m.group(1)), issue["body"].rstrip())
    sections = [f"<!-- engine-review-id: {issue['id']} -->", body]

    if issue.get("blocked_by"):
        sections.append("\n**Blocked by:** " + ", ".join(ref(d) for d in issue["blocked_by"]))
    if issue.get("epic"):
        sections.append(f"**Epic:** {ref(issue['epic'])}")

    children = [i for i in spec["issues"] if i.get("epic") == issue["id"]]
    if children:
        lines = ["\n## Tasks\n"]
        for child in children:
            lines.append(f"- [ ] {ref(child['id'])} {child['title']}")
        sections.append("\n".join(lines))

    sections.append(
        "\n---\n_Generated from `docs/research/reviews/2026-11-engine-review/issues.yaml` "
        "by `scripts/engine_review/bootstrap_github_project.py`._"
    )
    return "\n".join(sections) + "\n"


def create_issues(gh: Gh, spec: dict[str, Any], state: dict[str, int]) -> None:
    for issue in spec["issues"]:
        if issue["id"] in state:
            continue
        args = [
            "issue", "create",
            "--repo", gh.repo,
            "--title", issue["title"],
            "--body", placeholder_body(issue),
            "--milestone", issue["milestone"],
        ]
        for lbl in issue["labels"]:
            args.extend(["--label", lbl])
        url = gh.run(*args)
        if gh.dry_run:
            continue
        match = re.search(r"/issues/(\d+)$", url)
        if not match:
            raise SystemExit(f"could not parse issue number from: {url!r}")
        state[issue["id"]] = int(match.group(1))


def update_bodies(gh: Gh, spec: dict[str, Any], state: dict[str, int]) -> None:
    for issue in spec["issues"]:
        number = state.get(issue["id"])
        if number is None and not gh.dry_run:
            continue
        target = str(number) if number is not None else f"<{issue['id']}>"
        gh.run("issue", "edit", target, "--repo", gh.repo, "--body", final_body(issue, spec, state))


def project_owner(repo: str) -> str:
    return repo.split("/", 1)[0]


def setup_project(gh: Gh, spec: dict[str, Any], state: dict[str, int]) -> None:
    project = spec["project"]
    owner = project_owner(gh.repo)
    existing_number: int | None = None
    if not gh.dry_run:
        listing = gh.run("project", "list", "--owner", owner, "--format", "json", "--limit", "100", allow_fail=True)
        if listing:
            for p in json.loads(listing).get("projects", []):
                if p.get("title") == project["title"]:
                    existing_number = int(p["number"])
                    break
    if existing_number is None:
        out = gh.run("project", "create", "--owner", owner, "--title", project["title"], "--format", "json")
        existing_number = int(json.loads(out)["number"]) if out else 0
    number = str(existing_number)

    field_ids: dict[str, dict[str, Any]] = {}
    if not gh.dry_run:
        fields_json = gh.run("project", "field-list", number, "--owner", owner, "--format", "json")
        for f in json.loads(fields_json).get("fields", []):
            field_ids[f["name"]] = f
    for name, definition in project["fields"].items():
        if name in field_ids:
            continue
        gh.run(
            "project", "field-create", number,
            "--owner", owner,
            "--name", name,
            "--data-type", "SINGLE_SELECT",
            "--single-select-options", ",".join(definition["options"]),
        )
    if not gh.dry_run:
        fields_json = gh.run("project", "field-list", number, "--owner", owner, "--format", "json")
        field_ids = {f["name"]: f for f in json.loads(fields_json).get("fields", [])}
        project_id = json.loads(gh.run("project", "view", number, "--owner", owner, "--format", "json"))["id"]
    else:
        project_id = "<project-id>"

    for issue in spec["issues"]:
        issue_number = state.get(issue["id"])
        issue_url = f"https://github.com/{gh.repo}/issues/{issue_number if issue_number else '<' + issue['id'] + '>'}"
        item_json = gh.run("project", "item-add", number, "--owner", owner, "--url", issue_url, "--format", "json")
        item_id = json.loads(item_json)["id"] if item_json else "<item-id>"
        for field_name, value in issue.get("fields", {}).items():
            fdef = field_ids.get(field_name)
            option_id = "<option-id>"
            if fdef:
                option_id = next((o["id"] for o in fdef.get("options", []) if o["name"] == value), None)
                if option_id is None:
                    print(f"warning: option {value!r} not found for field {field_name!r}; skipping", file=sys.stderr)
                    continue
            gh.run(
                "project", "item-edit",
                "--project-id", project_id,
                "--id", item_id,
                "--field-id", fdef["id"] if fdef else f"<field:{field_name}>",
                "--single-select-option-id", option_id,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="id -> issue number map (idempotency)")
    parser.add_argument("--repo", default=None, help="owner/name; defaults to `gh repo view`")
    parser.add_argument("--dry-run", action="store_true", help="print gh commands without running them")
    parser.add_argument("--skip-project", action="store_true", help="create labels/milestones/issues only")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    repo = args.repo
    if repo is None:
        result = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                                capture_output=True, text=True, check=False)
        repo = result.stdout.strip() or "OWNER/REPO"
    gh = Gh(dry_run=args.dry_run, repo=repo)
    state = load_state(args.state)

    ensure_labels(gh, spec["labels"])
    ensure_milestones(gh, spec["milestones"])
    create_issues(gh, spec, state)
    save_state(args.state, state, args.dry_run)
    update_bodies(gh, spec, state)
    if not args.skip_project:
        setup_project(gh, spec, state)

    created = len(spec["issues"]) if args.dry_run else len(state)
    print(f"\n{'would create' if args.dry_run else 'tracked'} {created} issues across "
          f"{len(spec['milestones'])} milestones in {repo}")
    if not args.dry_run:
        print(f"state written to {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
