"""Build the ai-rag datasets: corpus, permissions, tenants, users, questions, red team, PDFs.

    uv run python build.py --profile mini
    uv run python build.py --profile mini --with-tickets
    uv run python build.py --profile full

Everything is deterministic (seed 42) and pinned to the revisions in sources.yaml.
Raw corpora are downloaded to the Hugging Face cache and never committed; build/ is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

DATASET_VERSION = "1.0.0"
SEED = 42
ROOT = Path(__file__).resolve().parent

ENTERPRISE_BENCH = "SJChen02/EnterpriseRAG-Bench"
TICKETS = "Tobi-Bueck/customer-support-tickets"
BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

MINI_SIZE = 10_000
DEV_FRACTION = 0.3
SUBSIDIARY_CLONES = 500
PDF_SLICE = {"total": 50, "scanned": 10, "tables": 10}
FULL_SOURCES = {"confluence", "google_drive", "jira", "github", "linear"}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

USER_PLAN = {
    "enterprise": {"employee": 6, "engineering": 5, "sales": 4, "hr": 2, "exec": 3},
    "subsidiary": {"employee": 3, "engineering": 3, "sales": 2, "hr": 1, "exec": 1},
}

FIRST_NAMES = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Daniel",
    "Nancy",
    "Matthew",
    "Lisa",
    "Anthony",
    "Betty",
    "Mark",
    "Sandra",
    "Steven",
    "Ashley",
    "Paul",
    "Kimberly",
    "Andrew",
    "Emily",
    "Joshua",
    "Donna",
    "Kenneth",
    "Michelle",
    "Kevin",
    "Carol",
    "Brian",
    "Amanda",
    "George",
    "Melissa",
    "Edward",
    "Deborah",
    "Ronald",
    "Stephanie",
    "Timothy",
    "Rebecca",
    "Jason",
    "Laura",
    "Jeffrey",
    "Sharon",
    "Ryan",
    "Cynthia",
    "Jacob",
    "Kathleen",
    "Gary",
    "Amy",
    "Nicholas",
    "Angela",
    "Eric",
    "Anna",
]
FAKE_FIRST = [
    "Alder",
    "Birch",
    "Cedar",
    "Dune",
    "Ember",
    "Fern",
    "Grove",
    "Harbor",
    "Indigo",
    "Juniper",
    "Kestrel",
    "Linden",
    "Maple",
    "Nova",
    "Onyx",
    "Piper",
    "Quill",
    "Rowan",
    "Sage",
    "Teal",
    "Umber",
    "Vale",
    "Wren",
    "Xander",
    "Yarrow",
    "Zephyr",
]
FAKE_LAST = [
    "Ashford",
    "Blackwood",
    "Calloway",
    "Dunmore",
    "Eastwick",
    "Fairbanks",
    "Garrity",
    "Hollis",
    "Ingram",
    "Jessop",
    "Kingsley",
    "Langdon",
    "Marlowe",
    "Northcott",
    "Oakes",
    "Pemberton",
    "Quinlan",
    "Rutherford",
    "Sterling",
    "Thackeray",
]


# ----------------------------------------------------------------------------- helpers
def norm_source(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", (value or "").strip().lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_sources(path: Path = ROOT / "sources.yaml") -> dict[str, str]:
    """Return {hf_repo_id: revision}. Accepts a flat mapping or one under a `sources:` key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = data.get("sources", data)
    missing = [r for r in (ENTERPRISE_BENCH,) if r not in data]
    if missing:
        raise SystemExit(f"sources.yaml has no pinned revision for: {missing}")
    return {str(k): str(v) for k, v in data.items()}


HEAD_CHARS = 1500


def header_emails(title: str, content: str, head_chars: int = HEAD_CHARS) -> set[str]:
    """Addresses in the head of a document: who a Gmail message is visible to."""
    return {e.lower() for e in EMAIL_RE.findall(title + "\n" + content[:head_chars])}


# ----------------------------------------------------------------------------- permissions
def load_acl_rules(path: Path = ROOT / "rules" / "acl_rules.yaml") -> dict:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    for src in rules["sources"].values():
        for r in src.get("rules", []):
            r["_re"] = re.compile(r["pattern"])
    return rules


def derive_acl(doc: dict, rules: dict) -> dict:
    source = doc["source_type"]
    spec = rules["sources"].get(source, rules["default"])
    head = doc["title"] + "\n" + doc["content"][: rules.get("head_chars", 1500)]
    allow_roles: list[str] = []
    allow_emails: list[str] = []
    rule = f"{source}:{spec['mode']}"
    visibility = "tenant"

    mode = spec["mode"]
    if mode == "roles":
        visibility, allow_roles = "roles", list(spec["roles"])
    elif mode == "principals":
        visibility = "principals"
        allow_emails = sorted(
            header_emails(
                doc["title"], doc["content"], rules.get("head_chars", HEAD_CHARS)
            )
        )
    elif mode == "keyword":
        for r in spec.get("rules", []):
            if r["_re"].search(head):
                visibility, allow_roles, rule = "roles", list(r["roles"]), r["name"]
                break
        else:
            visibility = spec.get("default", "tenant")
            rule = f"{source}:default"

    exec_ok = source not in rules.get("exec_sees_all_except", [])
    if visibility == "roles" and exec_ok and "exec" not in allow_roles:
        allow_roles.append("exec")
    return {
        "doc_id": doc["doc_id"],
        "tenant_id": doc["tenant_id"],
        "source_type": source,
        "visibility": visibility,
        "allow_roles": sorted(set(allow_roles)),
        "allow_emails": allow_emails,
        "rule": rule,
    }


def is_allowed(user: dict, acl: dict) -> bool:
    """The permission check every consumer repo must implement identically."""
    if user["tenant_id"] != acl["tenant_id"]:
        return False
    if acl["visibility"] == "tenant":
        return True
    if acl["visibility"] == "roles":
        return user["role"] in acl["allow_roles"]
    return user["email"].lower() in acl["allow_emails"]


# ----------------------------------------------------------------------------- tenants
GENERIC_LOCALS = {
    "info",
    "support",
    "sales",
    "admin",
    "noreply",
    "team",
    "hello",
    "oncall",
    "git",
    "security",
    "billing",
    "legal",
    "help",
    "contact",
    "press",
    "jobs",
    "careers",
    "ops",
    "dev",
    "eng",
    "finance",
    "marketing",
    "product",
    "it",
    "hr",
    "people",
    "alerts",
    "notifications",
    "no-reply",
    "root",
}


def discover_first_names(corpus: list[dict], domains: list[str]) -> set[str]:
    """First names of the company's people, read from its own email addresses (maya@redwood.ai)."""
    domains = {d.lower() for d in domains}
    names: Counter = Counter()
    for d in corpus:
        for email in EMAIL_RE.findall(d["title"] + "\n" + d["content"][:HEAD_CHARS]):
            local, _, domain = email.lower().partition("@")
            if domains and domain not in domains:
                continue
            first = re.split(r"[._-]", local)[0]
            if first.isalpha() and len(first) >= 3 and first not in GENERIC_LOCALS:
                names[first.capitalize()] += 1
    return set(names)


class Rewriter:
    """Deterministic, order-independent renaming of company, products, people, and emails."""

    def __init__(
        self,
        rules: dict,
        seed: int = SEED,
        extra_first_names: set[str] | frozenset = frozenset(),
    ):
        self.seed = seed
        self.company = sorted(rules.get("company") or [], key=len, reverse=True)
        self.products = sorted(rules.get("products") or [], key=len, reverse=True)
        self.company_to = rules.get("company_replacement", "Northwind Labs")
        self.product_to = rules.get("product_replacements") or ["NorthServe"]
        self.domains = {d.lower() for d in rules.get("domains") or []}
        names = "|".join(
            sorted(set(FIRST_NAMES) | set(extra_first_names), key=len, reverse=True)
        )
        self.person_re = re.compile(rf"\b({names})\s+([A-Z][A-Za-z']{{2,}})\b")

    def _pick(self, key: str, options: list[str]) -> str:
        h = int(hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest(), 16)
        return options[h % len(options)]

    def person(self, match: re.Match) -> str:
        full = match.group(0)
        return f"{self._pick('f:' + full, FAKE_FIRST)} {self._pick('l:' + full, FAKE_LAST)}"

    def email(self, match: re.Match) -> str:
        address = match.group(0)
        if self.domains and address.lower().partition("@")[2] not in self.domains:
            return address  # external addresses (git@github.com) stay
        return f"{self._pick('e:' + address.lower(), FAKE_FIRST).lower()}@northwind.example"

    def __call__(self, text: str) -> str:
        text = EMAIL_RE.sub(
            self.email, text
        )  # first, so company replacement cannot break domains
        for name in self.products:  # before company: product names often contain it
            text = text.replace(name, self._pick("p:" + name, self.product_to))
        for name in self.company:
            text = text.replace(name, self.company_to)
        return self.person_re.sub(self.person, text)


def make_subsidiary(
    corpus: list[dict], questions: list[dict], rewriter: Rewriter, rng: random.Random
):
    """Clone Confluence/Drive documents (referenced ones first) and rewrite matching questions."""
    eligible = [d for d in corpus if d["source_type"] in {"confluence", "google_drive"}]
    referenced = {i for q in questions for i in q["expected_doc_ids"]}
    first = [d for d in eligible if d["doc_id"] in referenced]
    rest = [d for d in eligible if d["doc_id"] not in referenced]
    rng.shuffle(rest)
    chosen = (sorted(first, key=lambda d: d["doc_id"]) + rest)[:SUBSIDIARY_CLONES]
    chosen_ids = {d["doc_id"] for d in chosen}

    clones = [
        {
            "doc_id": f"sub_{d['doc_id']}",
            "source_type": d["source_type"],
            "title": rewriter(d["title"]),
            "content": rewriter(d["content"]),
            "tenant_id": "subsidiary",
            "clone_of": d["doc_id"],
        }
        for d in chosen
    ]
    sub_questions = []
    for q in questions:
        exp = q["expected_doc_ids"]
        if exp and all(i in chosen_ids for i in exp):
            sub_questions.append(
                {
                    **q,
                    "question_id": f"sub_{q['question_id']}",
                    "question": rewriter(q["question"]),
                    "gold_answer": rewriter(q.get("gold_answer", "")),
                    "answer_facts": [rewriter(f) for f in q.get("answer_facts", [])],
                    "expected_doc_ids": [f"sub_{i}" for i in exp],
                    "tenant_id": "subsidiary",
                    "origin_question_id": q["question_id"],
                }
            )
    return clones, sub_questions


def build_users(
    corpus: list[dict], rng: random.Random, needed: set[str] | frozenset = frozenset()
) -> list[dict]:
    """30 users. Enterprise emails come from real Gmail headers so Gmail ACLs have principals.

    Addresses are chosen by greedy set cover over the Gmail documents that questions expect,
    so every expected document has at least one user who may read it; the rest are the
    most frequent addresses.
    """
    counts: Counter = Counter()
    wanted: dict[str, set[str]] = {}
    for d in corpus:
        if d["source_type"] == "gmail" and d["tenant_id"] == "enterprise":
            emails = header_emails(d["title"], d["content"])
            counts.update(emails)
            if d["doc_id"] in needed and emails:
                wanted[d["doc_id"]] = emails
    n_enterprise = sum(USER_PLAN["enterprise"].values())
    chosen: list[str] = []
    uncovered = dict(wanted)
    while uncovered and len(chosen) < n_enterprise:
        tally: Counter = Counter()
        for emails in uncovered.values():
            tally.update(emails)
        best = max(sorted(tally), key=lambda e: (tally[e], counts[e]))
        chosen.append(best)
        uncovered = {k: v for k, v in uncovered.items() if best not in v}
    if uncovered:
        print(
            f"WARNING: {len(uncovered)} expected Gmail documents cannot be covered by {n_enterprise} users"
        )
    chosen += [e for e, _ in counts.most_common() if e not in chosen]

    users, n = [], 0
    for tenant, plan in USER_PLAN.items():
        pool = list(chosen) if tenant == "enterprise" else []
        for role, count in plan.items():
            for _ in range(count):
                n += 1
                email = pool.pop(0) if pool else f"user{n:02d}@{tenant}.example"
                users.append(
                    {
                        "user_id": f"u{n:03d}",
                        "email": email,
                        "role": role,
                        "tenant_id": tenant,
                    }
                )
    return users


# ----------------------------------------------------------------------------- questions
def split_questions(questions: list[dict], rng: random.Random) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        by_type.setdefault(q["question_type"], []).append(q)
    out = []
    for qtype in sorted(by_type):
        group = by_type[qtype]
        rng.shuffle(group)
        n_dev = max(1, round(len(group) * DEV_FRACTION)) if len(group) > 1 else 0
        for i, q in enumerate(group):
            out.append(
                {
                    **q,
                    "tenant_id": "enterprise",
                    "split": "dev" if i < n_dev else "heldout",
                }
            )
    return sorted(out, key=lambda q: q["question_id"])


# ----------------------------------------------------------------------------- PDF slice
MAX_KEY_LEN = 40
MAX_CELL_CHARS = 600


def _table_rows(content: str) -> list[list[str]]:
    rows = [
        [c.strip() for c in ln.strip("| ").split("|")]
        for ln in content.splitlines()
        if ln.count("|") >= 2
    ]
    if len(rows) < 3:
        rows = [
            [c.strip() for c in ln.split("\t")]
            for ln in content.splitlines()
            if "\t" in ln
        ]
    if len(rows) < 3:
        rows = [
            [a.strip(), b.strip()]
            for ln in content.splitlines()
            if ":" in ln
            for a, b in [ln.split(":", 1)]
            if len(a.strip()) <= MAX_KEY_LEN
        ]
    return [r for r in rows if any(r)][:25]


def _usable_table(rows: list[list[str]]) -> bool:
    """At least 3 rows and no cell too tall for a PDF page (reportlab cannot split a cell)."""
    return len(rows) >= 3 and max(len(c) for r in rows for c in r) <= MAX_CELL_CHARS


def render_pdfs(corpus: list[dict], out_dir: Path, rng: random.Random) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
    from reportlab.platypus.doctemplate import LayoutError

    drive = sorted(
        (
            d
            for d in corpus
            if d["source_type"] == "google_drive" and d["tenant_id"] == "enterprise"
        ),
        key=lambda d: d["doc_id"],
    )
    if not drive:
        return []
    rng.shuffle(drive)
    with_tables = [d for d in drive if _usable_table(_table_rows(d["content"]))]
    picked_tables = with_tables[: PDF_SLICE["tables"]]
    remaining = [d for d in drive if d not in picked_tables]
    scanned = remaining[: PDF_SLICE["scanned"]]
    plain = remaining[PDF_SLICE["scanned"] : PDF_SLICE["total"] - len(picked_tables)]

    styles = getSampleStyleSheet()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _text_story(content: str) -> list:
        return [
            Paragraph(esc(para) or "&nbsp;", styles["BodyText"])
            for para in content.split("\n")
        ]

    for kind, docs in (
        ("tables", picked_tables),
        ("scanned", scanned),
        ("text", plain),
    ):
        for d in docs:
            path = out_dir / f"{d['doc_id']}.pdf"
            if kind == "scanned":
                w, h = 1240, 1754
                noise = Image.frombytes("L", (w, h), rng.randbytes(w * h))
                page = Image.new("L", (w, h), 245)
                draw = ImageDraw.Draw(page)
                font = ImageFont.load_default(size=22)
                y = 80
                lines = [d["title"], ""]
                for para in d["content"].splitlines():
                    while len(para) > 90:
                        cut = para.rfind(" ", 0, 90)
                        cut = cut if cut > 0 else 90
                        lines.append(para[:cut])
                        para = para[cut:].lstrip()
                    lines.append(para)
                for ln in lines:
                    if y > h - 80:
                        break
                    draw.text((90, y), ln, fill=30, font=font)
                    y += 32
                page = Image.blend(page, noise, 0.08).rotate(
                    rng.uniform(-0.8, 0.8), fillcolor=240
                )
                buf = io.BytesIO()
                page.save(buf, "PNG")
                buf.seek(0)
                c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
                c.drawImage(ImageReader(buf), 0, 0, width=letter[0], height=letter[1])
                c.save()
            else:
                doc = SimpleDocTemplate(
                    str(path), pagesize=letter, invariant=1, title=d["title"][:80]
                )
                story = [Paragraph(esc(d["title"]), styles["Title"])]
                if kind == "tables":
                    rows = _table_rows(d["content"])
                    width = max(len(r) for r in rows)
                    rows = [r + [""] * (width - len(r)) for r in rows]
                    cell = styles["BodyText"]
                    tbl = Table(
                        [[Paragraph(esc(c), cell) for c in r] for r in rows],
                        repeatRows=1,
                    )
                    tbl.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                            ]
                        )
                    )
                    story.append(tbl)
                else:
                    story.extend(_text_story(d["content"]))
                try:
                    doc.build(story)
                except LayoutError:
                    # Defensive: never let one odd document break the release build.
                    doc = SimpleDocTemplate(
                        str(path), pagesize=letter, invariant=1, title=d["title"][:80]
                    )
                    doc.build(
                        [Paragraph(esc(d["title"]), styles["Title"])]
                        + _text_story(d["content"])
                    )
            manifest.append({"doc_id": d["doc_id"], "file": path.name, "kind": kind})
    return manifest


# ----------------------------------------------------------------------------- tickets
PRIORITY_MAP = {
    "critical": "high",
    "urgent": "high",
    "high": "high",
    "medium": "medium",
    "normal": "medium",
    "low": "low",
    "very_low": "low",
    "very low": "low",
}


def build_tickets(rows: list[dict], out_dir: Path, rng: random.Random) -> dict:
    """English subset, deduplicated, 10 main queues, priority normalised, leakage-free splits."""
    from datasketch import MinHash, MinHashLSH

    def tags(r):
        return [str(r[k]) for k in sorted(r) if k.startswith("tag") and r.get(k)]

    rows = [
        r
        for r in rows
        if str(r.get("language", "en")).lower() == "en" and r.get("body")
    ]
    queues = [
        q
        for q, _ in Counter(r.get("queue") for r in rows if r.get("queue")).most_common(
            10
        )
    ]
    rows = [r for r in rows if r.get("queue") in queues]

    seen, uniq = set(), []
    for r in rows:
        key = hashlib.sha256(f"{r.get('subject', '')}|{r['body']}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    lsh = MinHashLSH(threshold=0.85, num_perm=64)
    sigs = []
    for i, r in enumerate(uniq):
        words = f"{r.get('subject', '')} {r['body']}".lower().split()
        m = MinHash(num_perm=64)
        for j in range(max(1, len(words) - 2)):
            m.update(" ".join(words[j : j + 3]).encode())
        sigs.append(m)
        lsh.insert(str(i), m)
    parent = list(range(len(uniq)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            parent[find(int(j))] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(uniq)):
        groups.setdefault(find(i), []).append(i)

    keys = sorted(groups)
    rng.shuffle(keys)
    cuts = {"train": 0.7, "val": 0.8, "dev": 0.9}
    total, acc, split_of = len(uniq), 0, {}
    for k in keys:
        frac = acc / total
        split = (
            "train"
            if frac < cuts["train"]
            else "val"
            if frac < cuts["val"]
            else "dev"
            if frac < cuts["dev"]
            else "heldout"
        )
        for i in groups[k]:
            split_of[i] = split
        acc += len(groups[k])

    out = []
    for i, r in enumerate(uniq):
        pr = PRIORITY_MAP.get(str(r.get("priority", "")).strip().lower(), "medium")
        out.append(
            {
                "ticket_id": f"tkt_{i:06d}",
                "subject": r.get("subject") or "",
                "body": r["body"],
                "answer": r.get("answer") or "",
                "type": r.get("type") or "",
                "queue": r["queue"],
                "priority": pr,
                "tags": tags(r),
                "split": split_of[i],
            }
        )
    counts = Counter(t["split"] for t in out)
    for split in ("train", "val", "dev", "heldout"):
        write_jsonl(out_dir / f"{split}.jsonl", (t for t in out if t["split"] == split))
    return {
        "queues": queues,
        "counts": dict(counts),
        "near_duplicate_groups": sum(len(g) > 1 for g in groups.values()),
    }


# ----------------------------------------------------------------------------- main build
def select_corpus(
    profile: str,
    ids: list[str],
    sources: list[str],
    needed: set[str],
    rng: random.Random,
    mini_size: int = MINI_SIZE,
) -> list[int]:
    """Return sorted row indices of the documents that belong to the profile."""
    index_of_needed = {i for i, d in enumerate(ids) if d in needed}
    if profile == "full":
        keep = {i for i, s in enumerate(sources) if s in FULL_SOURCES} | index_of_needed
    else:
        keep = set(index_of_needed)
        pool = [i for i in range(len(ids)) if i not in keep]
        rng.shuffle(pool)
        keep |= set(pool[: max(0, mini_size - len(keep))])
    return sorted(keep)


def build(
    profile: str,
    docs_ds,
    questions_raw: list[dict],
    out_root: Path,
    *,
    revisions: dict,
    rules_dir: Path = ROOT / "rules",
    mini_size: int = MINI_SIZE,
    tickets_rows: list[dict] | None = None,
    handwritten_dir: Path = ROOT,
) -> dict:
    rng = random.Random(SEED)
    out = out_root / profile
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ids = list(docs_ds["doc_id"])
    sources = [norm_source(s) for s in docs_ds["source_type"]]
    known = set(ids)

    # kept questions: every expected document exists in the source data
    kept = [
        q
        for q in questions_raw
        if all(i in known for i in q.get("expected_doc_ids") or [])
    ]
    questions = [
        {
            "question_id": q["question_id"],
            "question_type": q.get("question_type", ""),
            "source_types": [norm_source(s) for s in (q.get("source_types") or [])],
            "question": q["question"],
            "expected_doc_ids": list(q.get("expected_doc_ids") or []),
            "gold_answer": q.get("gold_answer") or "",
            "answer_facts": list(q.get("answer_facts") or []),
        }
        for q in kept
    ]
    needed = {i for q in questions for i in q["expected_doc_ids"]}

    idx = select_corpus(profile, ids, sources, needed, rng, mini_size)
    corpus = []
    for row, i in zip(docs_ds.select(idx), idx, strict=True):
        corpus.append(
            {
                "doc_id": row["doc_id"],
                "source_type": sources[i],
                "title": row.get("title") or "",
                "content": row.get("content") or "",
                "tenant_id": "enterprise",
            }
        )

    # tenants
    sub_rules = (
        yaml.safe_load(
            (rules_dir / "subsidiary_rewrite.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    first_names = discover_first_names(corpus, sub_rules.get("domains") or [])
    print(f"Discovered {len(first_names)} first names from company email addresses")
    clones, sub_questions = make_subsidiary(
        corpus, questions, Rewriter(sub_rules, extra_first_names=first_names), rng
    )
    all_docs = corpus + clones

    # permissions, users, questions
    acl_rules = load_acl_rules(rules_dir / "acl_rules.yaml")
    acl = [derive_acl(d, acl_rules) for d in all_docs]
    users = build_users(corpus, rng, needed)
    split = split_questions(questions, random.Random(SEED))
    split_by_id = {q["question_id"]: q["split"] for q in split}
    sub_split = [
        {**q, "split": split_by_id[q["origin_question_id"]]} for q in sub_questions
    ]

    for d in all_docs:
        write_json(out / "corpus" / d["source_type"] / f"{d['doc_id']}.json", d)
    write_jsonl(out / "acl.jsonl", acl)
    write_jsonl(out / "users.jsonl", users)
    write_jsonl(
        out / "questions" / "dev.jsonl", (q for q in split if q["split"] == "dev")
    )
    write_jsonl(
        out / "questions" / "heldout.jsonl",
        (q for q in split if q["split"] == "heldout"),
    )
    write_jsonl(out / "questions" / "subsidiary.jsonl", sub_split)

    # hand-written assets (copied, never generated)
    warnings = []
    probes = read_jsonl(handwritten_dir / "questions" / "probes.jsonl")
    write_jsonl(out / "questions" / "probes.jsonl", probes)
    gold = read_jsonl(handwritten_dir / "extraction_gold.jsonl")
    write_jsonl(out / "extraction_gold.jsonl", gold)
    red_src = handwritten_dir / "redteam"
    red = (
        [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(red_src.glob("*.json"))
        ]
        if red_src.exists()
        else []
    )
    for r in red:
        write_json(out / "redteam" / f"{r['doc_id']}.json", r)
    for name, n, need in (
        ("probes", len(probes), 30),
        ("redteam", len(red), 10),
        ("extraction_gold", len(gold), 25),
    ):
        if n < need:
            warnings.append(
                f"hand-written {name}: {n} found, {need} required for v1.0.0"
            )

    pdfs = render_pdfs(corpus, out / "pdf_slice", rng)
    write_jsonl(out / "pdf_slice" / "index.jsonl", pdfs)
    if not (sub_rules.get("company") or sub_rules.get("products")):
        warnings.append(
            "rules/subsidiary_rewrite.yaml has no company/product names; only people and emails were rewritten"
        )

    tickets_info = None
    if tickets_rows is not None:
        tickets_info = build_tickets(
            tickets_rows, out_root / "tickets", random.Random(SEED)
        )

    manifest = write_manifest(
        out_root,
        out,
        profile,
        revisions,
        {
            "documents": len(corpus),
            "subsidiary_documents": len(clones),
            "questions": len(questions),
            "subsidiary_questions": len(sub_questions),
            "users": len(users),
            "pdfs": len(pdfs),
            "tickets": tickets_info,
        },
    )
    for w in warnings:
        print("WARNING:", w)
    return manifest


def write_manifest(
    out_root: Path, out: Path, profile: str, revisions: dict, counts: dict
) -> dict:
    files = {}
    for base in (out, out_root / "tickets"):
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    files[p.relative_to(out_root).as_posix()] = sha256_file(p)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "profile": profile,
        "sources": revisions,
        "counts": counts,
        "files": files,
    }
    # Keep profiles from other builds listed in the same manifest.
    path = out_root / "MANIFEST.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("dataset_version") == DATASET_VERSION:
            merged = {**old.get("files", {}), **files}
            manifest["files"] = {
                k: v for k, v in merged.items() if (out_root / k).exists()
            }
            manifest["profiles"] = sorted(
                set(old.get("profiles", [old.get("profile")])) | {profile}
            )
    manifest.setdefault("profiles", [profile])
    write_json(path, manifest)
    return manifest


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profile", choices=["mini", "full"], default="mini")
    ap.add_argument("--with-tickets", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "build")
    ap.add_argument("--mini-size", type=int, default=MINI_SIZE)
    args = ap.parse_args()

    from datasets import load_dataset

    revisions = load_sources()
    rev = revisions[ENTERPRISE_BENCH]
    print(f"Loading {ENTERPRISE_BENCH}@{rev[:8]} ...")
    docs = load_dataset(ENTERPRISE_BENCH, "documents", split="test", revision=rev)
    questions = list(
        load_dataset(ENTERPRISE_BENCH, "questions", split="test", revision=rev)
    )

    tickets_rows = None
    if args.with_tickets:
        trev = revisions.get(TICKETS)
        if not trev:
            raise SystemExit(f"sources.yaml has no revision for {TICKETS}")
        tickets_rows = list(load_dataset(TICKETS, split="train", revision=trev))

    manifest = build(
        args.profile,
        docs,
        questions,
        args.out,
        revisions=revisions,
        mini_size=args.mini_size,
        tickets_rows=tickets_rows,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote {args.out / args.profile} and {args.out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
