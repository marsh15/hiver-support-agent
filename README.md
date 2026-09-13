# hiver-support-agent

AI support agent for **Spotify Cares**, built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (~3M tweets). Given the **first customer message of a thread**, the agent:

1. **Classifies** it into a 10-intent taxonomy induced from the data (`configs/intents.yaml`),
2. **Drafts a reply** grounded in how Spotify actually resolved similar issues
   (per-intent playbooks distilled from history + top-k retrieved real reply pairs, cited by thread id),
3. **Decides auto-handle vs escalate**, with an auditable reason
   (LLM proposal + deterministic guardrails that can only force escalation).

The full write-up — problem framing, baselines, failure analysis, and the mandatory
"what is misleading about my headline number" section — is in [REPORT.md](REPORT.md).

## Quick start (reproduce headline results in <15 min)

```bash
git clone <repo> && cd hiver-support-agent
make setup                              # venv + deps (~1 min)
export OPENAI_API_KEY=sk-...            # from https://platform.openai.com/api-keys
make eval report                        # all 4 systems over the 200-example golden set
```

- The brand subsample (**27,914 threads**) is committed at
  `data/committed/spotify_subsample.csv.gz` — you never need Kaggle.
- The retrieval index over the labeled corpus is committed at `index/` — eval
  needs no embedding spend. Every LLM call is disk-cached locally in `cache/`
  (gitignored), so re-runs on this machine are free; a grader's fresh,
  uncached eval costs ≈ **$3–5** on `gpt-4.1-mini` + `gpt-4.1`.
- Full pipeline from scratch (taxonomy → labels → index → playbooks → golden →
  eval): `make all`.

Try it live:

```bash
python demo.py "yall charged my account 8 times uhhh"
python demo.py "songs keep skipping on my daily mix, app is useless"
```

## Repo map

| path | what it is |
|---|---|
| `src/hiver_agent/` | the agent: `classify` → `draft` → `escalate` (+ `retrieval`, `llm` cache, `metrics`, `judge`, `baselines`) |
| `scripts/` | pipeline stages (data prep → taxonomy → labels → index → playbooks → golden → eval → report) |
| `configs/` | `intents.yaml` (taxonomy + always-escalate flags), `playbooks.yaml` (per-intent distilled playbooks) |
| `data/golden.csv` | the 200-example golden evaluation set (labels + `verified` flag) |
| `labeling/` | human verification sheets (golden, blind relabel, judge validation) |
| `index/` | committed embedding index over the labeled corpus |
| `results/` | metrics, predictions, confusion matrices, `SUMMARY.md` |
| `tests/test_core.py` | runnable logic checks (no network) — `make verify` |

## Architecture

![architecture](docs/architecture.html)

```
customer tweet ──▶ classify (few-shot LLM + 8 retrieved labeled neighbors)
                        │ intent + confidence + neighbor agreement
                        ▼
                 draft (gpt-4.1: intent playbook + 4 retrieved (msg→reply) exemplars)
                        │ reply ≤280 chars, no links/PII, grounding thread ids
                        ▼
                 escalate (LLM proposal ← guardrails: always-escalate intents,
                        │   low confidence, anger/legal regex, empty/invalid draft)
                        ▼
              auto-handle ────────────────── escalate to human (+ reason)
```

## Evaluation in one paragraph

200 hand-verified examples (`data/golden.csv`, sampling + labeling method in the
report). Intent: accuracy + macro-F1 + confusion. Escalation: escalate-class
P/R/F1 plus **auto-handle safety** — of the messages we auto-handled, the
fraction where auto-handling was actually right (the "can you trust it" metric).
Reply quality: an LLM judge (gpt-4.1-mini) scores groundedness / actionability /
tone / safety 1–5, validated against the author's own blind hand-scoring of 50
replies (`labeling/judge_sheet.csv` → `results/judge_agreement.json`). All
headline numbers carry bootstrap 95% CIs. Baselines: **trivial** (majority class
+ canned reply + always escalate), **simple** (TF-IDF logreg + per-intent
historical boilerplate + keyword escalation), and a **no-retrieval ablation**
(proves grounding matters).

## Citations

- Dataset: Scott Godwin (thoughtvector), *Customer Support on Twitter*, Kaggle.
- Model APIs: OpenAI (`gpt-4.1`, `gpt-4.1-mini`, `text-embedding-3-small`).
- Taxonomy seeds and escalation guardrail patterns are my own, induced from the
  data; nothing else was borrowed.
