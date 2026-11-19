# ai-rag-data: Shared Datasets

One repo that pins the public Hugging Face datasets ai-rag uses, adds the metadata the projects need (permissions, tenants, red-team documents, labels), and fixes the splits once, so no project tunes on another project's heldout set. It contains scripts and small derived files only; the corpora are downloaded at pinned revisions by `build.py`, and each tagged build is published once to S3 for the AWS-hosted projects.

**The AWS-hosted repos (`evalkit`, `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot`) use EnterpriseRAG-Bench only.** The ticket dataset is kept for `ticket-router-lora` and `llm-serving-lab`, which run off AWS.

## Sources

| Dataset | License | What it gives us | Used by |
|---------|---------|------------------|---------|
| [EnterpriseRAG-Bench](https://huggingface.co/datasets/SJChen02/EnterpriseRAG-Bench) (Onyx) | MIT | A simulated AI-inference company: about 512k documents from Slack, Gmail, Linear, Google Drive, HubSpot, meeting transcripts, GitHub, Jira, and Confluence, plus 500 questions in 10 categories (single-doc, multi-doc, conflicting information, and "info not found") with expected doc ids and gold answers | `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot`, `evalkit` |
| [customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | CC BY-NC 4.0 | About 62k English/German tickets with subject, body, agent answer, type, queue, priority, and tags | `ticket-router-lora`, `llm-serving-lab` only (off AWS) |
| [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (optional) | CDLA-Sharing-1.0 | About 27k synthetic requests across 27 intents | Optional robustness check for `ticket-router-lora` |

Cite each dataset in every repo's README that uses it.

## What this repo adds

| Asset | Used by | Spec |
|-------|---------|------|
| `corpus` profile `full` | `enterprise-rag-qa`, `workplace-copilot` | EnterpriseRAG-Bench Confluence, Drive, Jira, GitHub, and Linear documents, plus the Slack and Gmail documents any kept question references |
| `corpus` profile `mini` | all CI, laptops, the default AWS deployment | About 10k documents: every document referenced by a kept question plus a random sample, seed 42 |
| Tenants | `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot` | `enterprise`: the EnterpriseRAG-Bench company. `subsidiary`: 500 Confluence and Drive documents cloned with company, product, and people names rewritten (seed 42), so cross-tenant leakage shows up as a near-identical document from the wrong tenant |
| `acl_rules.yaml` + `acl.jsonl` | `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot` | Derived permissions per document: Gmail visible to sender and recipients only, HubSpot to `sales`, GitHub / Jira / Linear to `engineering`, Confluence and Drive by space or folder, Slack by channel; every document carries `tenant_id` |
| `users.jsonl` | `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot`, ai-rag-infra (Cognito seed) | 30 test users across both tenants with `user_id`, `email`, `role` (`employee`, `engineering`, `sales`, `hr`, `exec`), and `tenant_id` |
| `questions/` | `enterprise-rag-qa`, `workplace-copilot`, `evalkit` | EnterpriseRAG-Bench questions whose expected documents are all in the corpus, split `dev` / `heldout` (seed 42); the same questions rewritten for the `subsidiary` tenant; 30+ hand-written permission-probe and injection cases |
| `redteam/` | `enterprise-rag-qa`, `ops-desk-agents`, `workplace-copilot` | 10+ planted malicious documents (indirect prompt injection, fake instructions, data-exfiltration lures), marked so they never leak into normal eval |
| `pdf_slice/` | `enterprise-rag-qa` | 50 Drive documents rendered to PDF by `build.py` (10 as scanned images, 10 with tables) to exercise layout-aware parsing |
| `extraction_gold.jsonl` | `enterprise-rag-qa`, `evalkit` | 25 Drive / Confluence documents hand-labeled with owner, date, decisions, action items, and systems mentioned |
| `tickets/` | `ticket-router-lora`, `llm-serving-lab` | English subset of customer-support-tickets, deduplicated, 10 main queues, priority normalized to `low` / `medium` / `high`, splits `train` / `val` / `dev` / `heldout` with a near-duplicate check across splits |
| `MANIFEST.json` | all | `dataset_version`, the Hugging Face repo id and commit revision of every source, content hashes of every derived file |

## Rules

- **EnterpriseRAG-Bench is for retrieval and evaluation only.** Its authors ask that it stay out of training corpora (it carries a canary string), so no project fine-tunes on it; `ticket-router-lora` trains on tickets only
- **customer-support-tickets is CC BY-NC 4.0**: fine for a public, non-commercial project like ai-rag with attribution, not for anything commercial
- Every source is pinned to a Hugging Face commit revision in `MANIFEST.json`; never pull `main`
- Never edit derived files by hand: change `build.py` or the rules, bump `dataset_version`, rebuild
- Hand-written assets (`redteam/`, permission probes, `extraction_gold.jsonl`) are reviewed and listed in `QUALITY.md` with who labeled them and when
- Raw corpora are not committed; `build/` is git-ignored
- The S3 copy is write-once per version: `ai-rag-data/<version>/` is never overwritten; a change means a new version

## Build output

```
build/
├── mini/                    # or full/
│   ├── corpus/              # <source_type>/<doc_id>.json (text + metadata)
│   ├── pdf_slice/
│   ├── acl.jsonl
│   ├── users.jsonl
│   ├── questions/{dev,heldout,probes}.jsonl
│   ├── redteam/
│   └── extraction_gold.jsonl
├── tickets/                 # only with --with-tickets (`ticket-router-lora`, `llm-serving-lab`)
└── MANIFEST.json
```

## Build

```bash
uv sync
uv run python build.py --profile mini            # or --profile full; writes build/mini/
uv run python build.py --profile mini --with-tickets   # adds build/tickets/ for `ticket-router-lora` and `llm-serving-lab`
uv run python validate.py                        # schemas, hashes, tenant and split leakage
```

The full sequence, from creating the repo to publishing on S3, is in [Getting Started](#getting-started).

---

## Repository

This dataset is its own GitHub repo, **`ai-rag-data`**, part of the [ai-rag-labs](https://github.com/Lopukhovych/ai-rag-labs). Create it first and build it locally, but publish `v1.0.0` only after `ai-rag-infra` has deployed `AiRagFoundation` (the data bucket and the deploy role). Every project repo depends on it.

| | |
|---|---|
| Repo | `github.com/Lopukhovych/ai-rag-data` |
| Description | Pinned public datasets with derived permissions, tenants, and fixed splits for ai-rag |
| Topics | `datasets`, `rag`, `llm-evaluation`, `huggingface`, `ai-engineering` |
| License | MIT for the scripts and hand-written assets; each source keeps its own license (see Sources) |
| Releases | Tag `v<dataset_version>` (for example `v1.0.0`) whenever the build output changes; the tag publishes the build to S3 |

### Used by

| Repo | Uses |
|------|------|
| ai-rag-infra | `users.jsonl` (Cognito seed) |
| evalkit | `questions/`, `corpus` (oracle context), `extraction_gold.jsonl` |
| enterprise-rag-qa | `corpus`, `acl.jsonl`, `users.jsonl`, `questions/`, `redteam/`, `pdf_slice/`, `extraction_gold.jsonl` |
| ops-desk-agents | `corpus` (Jira, Linear, Slack), `users.jsonl`, `redteam/` |
| workplace-copilot | `corpus` (`mini`), `acl.jsonl`, `users.jsonl`, `questions/` |
| ticket-router-lora | `tickets/` |
| llm-serving-lab | `tickets/` (heldout) |
