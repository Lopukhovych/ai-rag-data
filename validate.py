"""Validate a build: schemas, hashes, permissions, tenant and split leakage.

    uv run python validate.py                       # build/, every profile in MANIFEST.json
    uv run python validate.py --allow-missing-handwritten
Exit code 1 if any error is found; warnings do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from build import ROOT, is_allowed, read_jsonl, sha256_file
from models import (
    AclRecord,
    Doc,
    ExtractionGold,
    Probe,
    Question,
    RedteamDoc,
    Ticket,
    User,
)

MIN_HANDWRITTEN = {"probes": 30, "redteam": 10, "extraction_gold": 25}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_schema(rows: list[dict], model, label: str, rep: Report) -> list:
    out = []
    for i, row in enumerate(rows):
        try:
            out.append(model.model_validate(row))
        except ValidationError as e:
            rep.error(f"{label}[{i}]: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
            if len(rep.errors) > 50:
                break
    return out


def validate_profile(
    out_root: Path, profile: str, rep: Report, allow_missing_handwritten: bool
) -> None:
    base = out_root / profile
    if not base.exists():
        rep.error(f"{profile}: directory {base} missing")
        return

    docs = []
    for p in sorted((base / "corpus").rglob("*.json")):
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    parsed_docs = check_schema(docs, Doc, f"{profile}/corpus", rep)
    ids = [d.doc_id for d in parsed_docs]
    if len(ids) != len(set(ids)):
        rep.error(f"{profile}: duplicate doc_id in corpus")
    enterprise = {d.doc_id for d in parsed_docs if d.tenant_id == "enterprise"}
    subsidiary = {d.doc_id for d in parsed_docs if d.tenant_id == "subsidiary"}

    # tenants: no shared ids, clones really differ from their originals
    if enterprise & subsidiary:
        rep.error(f"{profile}: doc ids shared across tenants")
    by_id = {d.doc_id: d for d in parsed_docs}
    unchanged = 0
    for d in parsed_docs:
        if d.tenant_id == "subsidiary":
            if d.clone_of not in enterprise:
                rep.error(
                    f"{profile}: clone {d.doc_id} points at unknown original {d.clone_of}"
                )
            elif by_id[d.clone_of].content == d.content:
                unchanged += 1
    if unchanged:
        rep.warn(
            f"{profile}: {unchanged} subsidiary clones are identical to their original (rewrite rules matched nothing)"
        )

    # permissions
    acl = check_schema(read_jsonl(base / "acl.jsonl"), AclRecord, f"{profile}/acl", rep)
    acl_ids = {a.doc_id for a in acl}
    if acl_ids != set(ids):
        rep.error(
            f"{profile}: acl.jsonl and corpus differ ({len(acl_ids ^ set(ids))} ids)"
        )
    tenant_of = {d.doc_id: d.tenant_id for d in parsed_docs}
    for a in acl:
        if tenant_of.get(a.doc_id) != a.tenant_id:
            rep.error(f"{profile}: acl tenant mismatch for {a.doc_id}")
            break
    empty = sum(1 for a in acl if a.visibility == "principals" and not a.allow_emails)
    if empty:
        rep.warn(
            f"{profile}: {empty} Gmail documents have no addresses in their header (visible to nobody)"
        )

    # users
    users = check_schema(
        read_jsonl(base / "users.jsonl"), User, f"{profile}/users", rep
    )
    if len(users) != 30:
        rep.error(f"{profile}: users.jsonl has {len(users)} users, expected 30")
    if len({u.user_id for u in users}) != len(users) or len(
        {u.email for u in users}
    ) != len(users):
        rep.error(f"{profile}: duplicate user ids or emails")
    if {u.tenant_id for u in users} != {"enterprise", "subsidiary"}:
        rep.error(f"{profile}: users must cover both tenants")
    user_ids = {u.user_id for u in users}

    # questions and splits
    qdir = base / "questions"
    dev = check_schema(read_jsonl(qdir / "dev.jsonl"), Question, f"{profile}/dev", rep)
    held = check_schema(
        read_jsonl(qdir / "heldout.jsonl"), Question, f"{profile}/heldout", rep
    )
    sub = check_schema(
        read_jsonl(qdir / "subsidiary.jsonl"), Question, f"{profile}/subsidiary", rep
    )
    if {q.question_id for q in dev} & {q.question_id for q in held}:
        rep.error(f"{profile}: question ids appear in both dev and heldout")
    if any(q.split != "dev" for q in dev) or any(q.split != "heldout" for q in held):
        rep.error(f"{profile}: split field disagrees with file name")
    split_of = {q.question_id: q.split for q in dev + held}
    for q in dev + held:
        for i in q.expected_doc_ids:
            if i not in enterprise:
                rep.error(
                    f"{profile}: {q.question_id} expects {i}, not in the enterprise corpus"
                )
                break
    for q in sub:
        if split_of.get(q.origin_question_id) != q.split:
            rep.error(
                f"{profile}: {q.question_id} split differs from its origin question"
            )
        if any(i not in subsidiary for i in q.expected_doc_ids):
            rep.error(
                f"{profile}: {q.question_id} expects documents outside the subsidiary tenant"
            )

    # permission sanity: every question's expected docs must be readable by at least one user
    acl_by_id = {a.doc_id: a.model_dump() for a in acl}
    user_dicts = [u.model_dump() for u in users]
    for q in dev + held + sub:
        for i in q.expected_doc_ids:
            rec = acl_by_id.get(i)
            if rec and not any(
                is_allowed(u, rec) for u in user_dicts if u["tenant_id"] == q.tenant_id
            ):
                rep.warn(
                    f"{profile}: nobody can read {i} (expected by {q.question_id})"
                )
                break

    # hand-written assets
    probes = check_schema(
        read_jsonl(qdir / "probes.jsonl"), Probe, f"{profile}/probes", rep
    )
    for p in probes:
        if p.user_id not in user_ids:
            rep.error(f"{profile}: probe {p.probe_id} uses unknown user {p.user_id}")
    gold = check_schema(
        read_jsonl(base / "extraction_gold.jsonl"),
        ExtractionGold,
        f"{profile}/extraction_gold",
        rep,
    )
    for g in gold:
        if by_id.get(g.doc_id) is None or by_id[g.doc_id].source_type not in {
            "google_drive",
            "confluence",
        }:
            rep.error(
                f"{profile}: extraction_gold {g.doc_id} is not a Drive/Confluence document in the corpus"
            )
    red = []
    for p in sorted((base / "redteam").glob("*.json")):
        red.append(json.loads(p.read_text(encoding="utf-8")))
    red_parsed = check_schema(red, RedteamDoc, f"{profile}/redteam", rep)
    if {r.doc_id for r in red_parsed} & set(ids):
        rep.error(f"{profile}: red-team documents leaked into the normal corpus")
    for name, n in (
        ("probes", len(probes)),
        ("redteam", len(red_parsed)),
        ("extraction_gold", len(gold)),
    ):
        if n < MIN_HANDWRITTEN[name]:
            msg = f"{profile}: {name} has {n}, v1.0.0 needs {MIN_HANDWRITTEN[name]}"
            (rep.warn if allow_missing_handwritten else rep.error)(msg)

    # PDFs
    index = read_jsonl(base / "pdf_slice" / "index.jsonl")
    kinds = Counter(r["kind"] for r in index)
    for r in index:
        if not (base / "pdf_slice" / r["file"]).exists():
            rep.error(f"{profile}: pdf {r['file']} missing")
    if index and (len(index) != 50 or kinds["scanned"] != 10 or kinds["tables"] != 10):
        rep.warn(
            f"{profile}: pdf_slice is {dict(kinds)} (target 50 total, 10 scanned, 10 tables)"
        )


def validate_tickets(out_root: Path, rep: Report) -> None:
    base = out_root / "tickets"
    rows: dict[str, list] = {}
    for split in ("train", "val", "dev", "heldout"):
        rows[split] = check_schema(
            read_jsonl(base / f"{split}.jsonl"), Ticket, f"tickets/{split}", rep
        )
    seen: dict[str, str] = {}
    for split, items in rows.items():
        for t in items:
            key = f"{t.subject}|{t.body}"
            if key in seen and seen[key] != split:
                rep.error(f"tickets: identical ticket in {seen[key]} and {split}")
                return
            seen[key] = split
    if len({q for items in rows.values() for q in (t.queue for t in items)}) > 10:
        rep.error("tickets: more than 10 queues")


def validate_manifest(out_root: Path, rep: Report) -> dict:
    path = out_root / "MANIFEST.json"
    if not path.exists():
        rep.error("MANIFEST.json missing")
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "dataset_version" not in manifest or not manifest.get("sources"):
        rep.error("MANIFEST.json lacks dataset_version or sources")
    for rel, digest in manifest.get("files", {}).items():
        p = out_root / rel
        if not p.exists():
            rep.error(f"manifest lists missing file {rel}")
        elif sha256_file(p) != digest:
            rep.error(f"hash mismatch: {rel}")
    for repo, rev in manifest.get("sources", {}).items():
        if rev in ("main", "master", "") or len(rev) < 7:
            rep.error(f"source {repo} is not pinned to a commit ({rev!r})")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--allow-missing-handwritten", action="store_true")
    args = ap.parse_args()

    rep = Report()
    manifest = validate_manifest(args.build, rep)
    for profile in manifest.get("profiles", []):
        validate_profile(args.build, profile, rep, args.allow_missing_handwritten)
    if (args.build / "tickets").exists():
        validate_tickets(args.build, rep)

    for w in rep.warnings:
        print("WARNING:", w)
    for e in rep.errors:
        print("ERROR:", e)
    print(f"{len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
