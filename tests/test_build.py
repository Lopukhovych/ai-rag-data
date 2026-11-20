import json
from pathlib import Path

import pytest

import build as b
import validate as v
from tests import fakes

REVISIONS = {b.ENTERPRISE_BENCH: "a" * 40}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    hand = tmp / "hand"
    fakes.write_handwritten(hand)
    docs = fakes.make_docs()
    questions = fakes.make_questions(docs)
    manifest = b.build(
        "mini",
        docs,
        questions,
        tmp / "build",
        revisions=REVISIONS,
        mini_size=300,
        tickets_rows=fakes.make_tickets(),
        handwritten_dir=hand,
    )
    return tmp / "build", manifest, docs, questions


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_validate_passes(built):
    out, *_ = built
    rep = v.Report()
    manifest = v.validate_manifest(out, rep)
    for profile in manifest["profiles"]:
        v.validate_profile(out, profile, rep, allow_missing_handwritten=True)
    v.validate_tickets(out, rep)
    assert rep.errors == []


def test_corpus_contains_every_expected_document_and_has_target_size(built):
    out, manifest, _, _ = built
    ids = {p.stem for p in (out / "mini" / "corpus").rglob("*.json")}
    for q in rows(out / "mini" / "questions" / "dev.jsonl") + rows(
        out / "mini" / "questions" / "heldout.jsonl"
    ):
        assert set(q["expected_doc_ids"]) <= ids
    assert manifest["counts"]["documents"] == 300


def test_question_with_missing_document_is_dropped(built):
    out, *_ = built
    all_q = rows(out / "mini" / "questions" / "dev.jsonl") + rows(
        out / "mini" / "questions" / "heldout.jsonl"
    )
    assert "qst_bad" not in {q["question_id"] for q in all_q}
    assert len(all_q) == 60


def test_splits_are_disjoint_and_stratified(built):
    out, *_ = built
    dev = {q["question_id"] for q in rows(out / "mini" / "questions" / "dev.jsonl")}
    held = {
        q["question_id"] for q in rows(out / "mini" / "questions" / "heldout.jsonl")
    }
    assert dev and held and not dev & held
    assert 0.2 < len(dev) / (len(dev) + len(held)) < 0.4


def test_build_is_deterministic(built, tmp_path):
    _out, manifest, docs, questions = built
    hand = tmp_path / "hand"
    fakes.write_handwritten(hand)
    again = b.build(
        "mini",
        docs,
        questions,
        tmp_path / "build",
        revisions=REVISIONS,
        mini_size=300,
        tickets_rows=fakes.make_tickets(),
        handwritten_dir=hand,
    )
    assert again["files"] == manifest["files"]


def test_users(built):
    out, *_ = built
    users = rows(out / "mini" / "users.jsonl")
    assert len(users) == 30
    assert {u["tenant_id"] for u in users} == {"enterprise", "subsidiary"}
    assert {u["role"] for u in users} == set(b.USER_PLAN["enterprise"])


def test_acl_roles_and_gmail_principals(built):
    out, *_ = built
    acl = rows(out / "mini" / "acl.jsonl")
    by_src = {}
    for a in acl:
        by_src.setdefault(a["source_type"], []).append(a)
    assert all(a["allow_roles"] == ["engineering", "exec"] for a in by_src["github"])
    assert all(a["allow_roles"] == ["exec", "sales"] for a in by_src["hubspot"])
    assert all(
        a["visibility"] == "principals" and a["allow_emails"] for a in by_src["gmail"]
    )
    assert all("exec" not in a["allow_roles"] for a in by_src["gmail"])
    assert any(a["rule"] == "space-hr" for a in by_src["confluence"])
    assert any(a["rule"] == "slack-hr" for a in by_src["slack"])


def test_is_allowed():
    acl = {
        "tenant_id": "enterprise",
        "visibility": "roles",
        "allow_roles": ["hr"],
        "allow_emails": [],
    }
    assert b.is_allowed({"tenant_id": "enterprise", "role": "hr", "email": "a@x"}, acl)
    assert not b.is_allowed(
        {"tenant_id": "enterprise", "role": "sales", "email": "a@x"}, acl
    )
    assert not b.is_allowed(
        {"tenant_id": "subsidiary", "role": "hr", "email": "a@x"}, acl
    )
    mail = {
        "tenant_id": "enterprise",
        "visibility": "principals",
        "allow_roles": [],
        "allow_emails": ["a@x"],
    }
    assert b.is_allowed(
        {"tenant_id": "enterprise", "role": "exec", "email": "A@x"}, mail
    )
    assert not b.is_allowed(
        {"tenant_id": "enterprise", "role": "exec", "email": "b@x"}, mail
    )


def test_subsidiary_clones_are_rewritten_and_isolated(built):
    out, *_ = built
    clones = [
        json.loads(p.read_text()) for p in (out / "mini" / "corpus").rglob("sub_*.json")
    ]
    assert clones
    orig_dir = out / "mini" / "corpus"
    for c in clones[:20]:
        orig = json.loads(next(orig_dir.rglob(f"{c['clone_of']}.json")).read_text())
        assert c["tenant_id"] == "subsidiary" and orig["tenant_id"] == "enterprise"
        assert "Mary Smith" not in c["content"]
        assert "@acme.example" not in c["content"]
    sub_q = rows(out / "mini" / "questions" / "subsidiary.jsonl")
    assert sub_q and all(
        i.startswith("sub_") for q in sub_q for i in q["expected_doc_ids"]
    )
    assert all("Mary Smith" not in q["question"] for q in sub_q)


def test_rewriter_is_deterministic_and_order_independent():
    rules = {
        "company": ["Acme Cloud", "Acme"],
        "products": ["AcmeServe"],
        "company_replacement": "Northwind",
    }
    r1, r2 = b.Rewriter(rules), b.Rewriter(rules)
    text = "Mary Smith runs Acme Cloud and AcmeServe."
    assert r1(text) == r2(text)
    assert "Acme" not in r1(text) and "Mary Smith" not in r1(text)
    assert r1("Mary Smith") == r2("x Mary Smith")[2:]


def test_pdf_slice(built):
    out, *_ = built
    index = rows(out / "mini" / "pdf_slice" / "index.jsonl")
    assert index
    assert {r["kind"] for r in index} <= {"tables", "scanned", "text"}
    for r in index:
        assert (out / "mini" / "pdf_slice" / r["file"]).read_bytes().startswith(b"%PDF")


def test_tickets(built):
    out, *_ = built
    splits = {
        s: rows(out / "tickets" / f"{s}.jsonl")
        for s in ("train", "val", "dev", "heldout")
    }
    all_t = [t for items in splits.values() for t in items]
    assert {t["priority"] for t in all_t} <= {"low", "medium", "high"}
    assert len({t["queue"] for t in all_t}) <= 10
    keys = [(t["subject"], t["body"]) for t in all_t]
    assert len(keys) == len(set(keys))  # exact duplicates removed
    assert all(splits[s] for s in splits)


def test_manifest_pins_sources_and_hashes(built):
    _out, manifest, *_ = built
    assert manifest["dataset_version"] == "1.0.0"
    assert manifest["sources"][b.ENTERPRISE_BENCH] == "a" * 40
    assert "mini/acl.jsonl" in manifest["files"]


def test_validate_catches_tampering(built, tmp_path):
    out, *_ = built
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    (copy / "mini" / "acl.jsonl").write_text("")
    rep = v.Report()
    v.validate_manifest(copy, rep)
    assert any("hash mismatch" in e for e in rep.errors)


def test_load_sources_accepts_both_shapes(tmp_path):
    flat = tmp_path / "a.yaml"
    flat.write_text(f"{b.ENTERPRISE_BENCH}: {'b' * 40}\n")
    nested = tmp_path / "b.yaml"
    nested.write_text(f"sources:\n  {b.ENTERPRISE_BENCH}: {'b' * 40}\n")
    assert b.load_sources(flat) == b.load_sources(nested)
    with pytest.raises(SystemExit):
        empty = tmp_path / "c.yaml"
        empty.write_text("sources: {}\n")
        b.load_sources(empty)


def test_full_profile_selects_expected_sources():
    ids = [f"d{i}" for i in range(6)]
    sources = ["slack", "confluence", "jira", "gmail", "hubspot", "github"]
    import random

    idx = b.select_corpus("full", ids, sources, {"d3"}, random.Random(1))
    assert [ids[i] for i in idx] == ["d1", "d2", "d3", "d5"]


@pytest.mark.parametrize("profile", ["mini"])
def test_repo_rules_files_exist(profile):
    assert (Path(b.ROOT) / "rules" / "acl_rules.yaml").exists()
    assert (Path(b.ROOT) / "rules" / "subsidiary_rewrite.yaml").exists()


def test_rewriter_keeps_external_emails_and_renames_company_domain():
    rules = {
        "company": ["Redwood"],
        "company_replacement": "Northwind",
        "domains": ["redwood.ai"],
    }
    out = b.Rewriter(rules)(
        "Mail maya@redwood.ai or git@github.com about Redwood Inference."
    )
    assert "redwood" not in out.lower()
    assert "git@github.com" in out
    assert "Northwind Inference" in out


def test_rewriter_renames_discovered_names_with_apostrophes():
    r = b.Rewriter({"domains": ["redwood.ai"]}, extra_first_names={"Liam", "Aisha"})
    out = r("Liam O'Conn pairs with Aisha Patel.")
    assert "Liam" not in out and "Aisha" not in out and "Conn" not in out


def test_discover_first_names_from_company_emails():
    corpus = [
        {
            "title": "t",
            "content": "From: maya.chen@redwood.ai\nTo: support@redwood.ai, bob@other.com",
        },
        {"title": "t", "content": "cc: aisha@redwood.ai"},
    ]
    assert b.discover_first_names(corpus, ["redwood.ai"]) == {"Maya", "Aisha"}


def test_users_cover_every_expected_gmail_document():
    corpus = []
    needed = set()
    for i in range(
        40
    ):  # 40 messages, each between two unique addresses: needs several users
        corpus.append(
            {
                "doc_id": f"g{i}",
                "source_type": "gmail",
                "tenant_id": "enterprise",
                "title": "m",
                "content": f"From: a{i}@x.com\nTo: b{i // 2}@x.com",
            }
        )
        needed.add(f"g{i}")
    import random

    users = b.build_users(corpus, random.Random(1), needed)
    rules = b.load_acl_rules()
    enterprise = [u for u in users if u["tenant_id"] == "enterprise"]
    for d in corpus:
        acl = b.derive_acl(d, rules)
        assert any(b.is_allowed(u, acl) for u in enterprise), d["doc_id"]
    assert len(users) == 30
