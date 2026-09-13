# REPORT — AI Support Agent for Spotify Cares

*Hiver SDE intern take-home. All numbers regenerate via `make eval report`; this
report cites `results/SUMMARY.md` as its source of truth.*

## 1. Problem framing

**The job.** Spotify Cares answers thousands of public @-mentions a day. The
first reply sets the tone for the whole thread. I built a **turn-1 triage
agent**: it reads the first inbound customer tweet and must (a) classify the
issue, (b) draft the public first reply, (c) decide auto-handle vs escalate —
with a stated reason.

**What "good" means for this brand.**

1. **Safe first**: a wrong public reply is costlier than a slow one. The
   trust metric is *auto-handle safety* — of the tweets we auto-handled, the
   fraction where auto-handling was actually correct. Escalation recall beats
   escalation precision.
2. **Grounded, not creative**: the reply may only claim what Spotify
   historically claims for that issue. No invented policies, no public refund
   promises, no PII requests beyond "please DM us".
3. **On-brand**: short (≤280 chars), no links, no @-mentions in the draft,
   Spotify's concise-friendly voice.
4. **Useful triage**: the intent label and escalation reason must be
   explainable to a human taking over.

**What I chose not to build** (and why):

- **Multi-turn conversation state** — evaluating a stateful agent needs
  dialogue-level ground truth this dataset barely supports; turn-1 triage is
  the well-defined 80% case.
- **Banking77** (the optional second dataset) — a second domain would halve the
  depth of the evaluation story for zero addition to the core claim.
- **Fine-tuned models** — the few-shot + retrieval agent is stronger per dollar
  at this data scale and keeps every decision inspectable in a prompt.
- **A web UI** — `demo.py` demonstrates the full pipeline; a chat UI proves
  nothing about quality.

## 2. System

- **Taxonomy**: 10 intents induced from the data (LLM clustering of a 2k sample
  → hand-consolidated; `configs/intents.yaml`). `account_security` is flagged
  always-escalate.
- **Classifier**: `gpt-4.1-mini` with the intent definitions + 8 retrieved
  labeled neighbors in the prompt; returns intent + self-assessed confidence,
  cross-checked against neighbor-label agreement.
- **Drafter**: `gpt-4.1` with a per-intent **playbook** distilled from ≤200
  historical brand replies + 4 retrieved (customer msg → real brand reply)
  exemplars, cited by thread id. Deterministic post-check strips links/
  mentions and enforces ≤280 chars.
- **Escalation**: LLM proposes auto/escalate with a reason; deterministic
  guardrails can only *force* escalation (always-escalate intents, low
  confidence, anger/legal regex, empty draft) — never suppress it.

## 3. Results vs baselines

> Auto-filled table: `results/SUMMARY.md` (regenerates with `make report`).

<!-- RESULTS_TABLE -->

**Baselines.** *Trivial*: majority-class intent + one canned "please DM us"
reply + always escalate. *Simple*: TF-IDF logistic regression intent classifier
+ per-intent most-common historical reply + keyword-blacklist escalation.
*Ablation*: the full agent with retrieval removed — isolates how much the
grounding retrieval contributes.

**Reading the numbers.**

<!-- RESULTS_READING -->

## 4. Evaluation setup (the proof)

- **Golden set**: 200 examples, stratified by induced intent with ≥8 per intent
  (min-count intents filled from the remainder) from the 6k-row labeled pool,
  seed 13. Labels: LLM pre-labels that I then personally verified/corrected
  tweet-by-tweet against the real thread (`labeling/golden_sheet.html`); the
  `verified` flag in `data/golden.csv` marks this. Escalation ground truth is
  my judgment after reading the thread, with the brand's *actual historical
  behavior* (notably DM-redirect) shown as evidence — not the heuristic's own
  output.
- **Labeler self-agreement**: I re-labeled 30 examples blind (shuffled,
  labels hidden, `labeling/relabel_30.csv`). Agreement + kappa:
  <!-- SELF_AGREEMENT -->
- **LLM judge**: `gpt-4.1-mini`, temperature 0, scores groundedness /
  actionability / tone / safety 1–5 (pass = all ≥4). The judge sees the
  playbook and the same exemplar pool the drafter had, so "grounded" is
  checkable, not vibes.
- **Judge validation**: I hand-scored 50 drafted replies blind on the same
  rubric (`labeling/judge_sheet.csv`). Judge–human agreement:
  <!-- JUDGE_AGREEMENT -->
- **Uncertainty**: every headline number carries a bootstrap 95% CI (2,000
  resamples, seed 42).

## 5. Failure analysis (top 5 modes)

<!-- FAILURE_ANALYSIS -->

## 6. What is misleading about my headline number?

The honest list, most damaging first:

1. **Retrieval leakage.** Golden-set tweets can retrieve their *own* historical
   thread as an exemplar — the agent may be parroting the labeled answer rather
   than generalizing. Mitigation reported separately: a leakage-free subset
   metric excluding near-exact matches (see `results/`), but the headline
   grounding/judge numbers include leakage.
2. **My labels are the ground truth — and I built the system.** One annotator,
   not blind to the system's behavior, with a stake in it looking good. The
   30-example self-agreement audit bounds *intra*-rater noise, not bias.
3. **Judge leniency toward fluent text.** The judge shares a model family with
   the drafter; LLM judges systematically prefer LLM-style prose. My 50-reply
   validation quantifies agreement, but 50 examples from one human is thin
   evidence — the kappa CI is wide.
4. **Escalation ground truth is DM-redirect-tinted.** ~37% of historical first
   replies redirect to DM; my "should escalate" labels lean on that outcome, so
   a system that escalates often looks aligned with history even if a better
   agent would auto-resolve more.
5. **2017 data, one brand, English-only.** Intent boundaries, response norms
   and even the product (no audiobooks, no AI DJ) are six years stale; macro-F1
   is dominated by the `other` bucket's behavior; and every number is a point
   estimate on n=200 with wide CIs on rare intents.

## 7. With one more week

1. **Second annotator** on the golden set + judge sheet → inter-annotator
   kappa, and a leakage-controlled eval split (golden tweets' own threads
   banned from retrieval).
2. **Threshold tuning on a held-out split** — confidence and guardrail
   thresholds are currently set by judgment, not by an ROC over a tuning set;
   I'd sweep them for a target auto-handle safety ≥0.95.
3. **Multi-turn**: thread-aware triage using the 2-level thread head already
   captured, with a "context changed the intent" eval slice.
4. **A distilled cheap classifier** (logreg/MiniLM on the 6k induced labels)
   as a candidate *replacement* for the LLM classifier — if it matches quality
   at 100x lower cost, the LLM moves to judge-only.
5. **Online failure mining**: cluster judge-failed replies by rationale to
   find the next failure mode systematically instead of by anecdote.

## 8. Decision log (the non-obvious calls)

- **Spotify Cares** over Apple/Amazon: richest *resolvable* threads per unit of
  noise; taxonomy is human-scale.
- **Turn-1 only**: one input → one prediction makes ground truth, baselines and
  CIs clean; multi-turn deferred deliberately.
- **10 intents + `other`**, induced-then-hand-consolidated: LLM clustering
  alone produced fuzzy duplicates; hand consolidation is where the taxonomy
  became defensible.
- **Few-shot retrieval classifier, not fine-tuning**: 6k labels is enough for
  retrieval shots but thin for a reliable fine-tune; and prompt = policy you
  can read in an interview.
- **Self-reported confidence × neighbor agreement** as the uncertainty signal:
  neither alone is calibrated; the combination is the cheap proxy we have.
- **Playbook + exemplars, not exemplars alone**: playbooks stabilize tone
  across intents where retrieval neighbors are inconsistent.
- **Guardrails can only force escalation**: asymmetry is the whole design —
  automation may decline, never overrule a safety rule.
- **Auto-handle safety as the headline trust metric**: P/R/F1 on "escalate"
  hides the operational question ("how often are we confidently wrong in
  public?").
- **LLM-judge ≠ drafter model** (4.1-mini vs 4.1) and judge sees the exemplar
  pool: reduces self-preference and makes "grounded" auditable.
- **Golden labels = my verification of LLM pre-labels**, disclosed as such:
  pure hand-labeling of 200 tweets was the same hours with worse consistency;
  the honest framing is in §6.
- **All LLM calls disk-cached and the cache committed**: graders reproduce
  numbers with zero API spend and zero variance.
- **Committed 27,914-thread subsample**: the assignment says a subsample is
  expected; full-data runs stay possible via `make data`.
- **No LangChain/vector DB**: the pipeline is ~700 lines a reviewer can read in
  the live interview; numpy cosine is enough at this scale.
- **Banking77 skipped**: depth over breadth; logged, not hidden.
- **Bootstrap CIs on everything**: with n=200, a 3-point accuracy gap between
  systems can be noise; CIs force the report to say so.

## 9. Reproducing

See README quickstart. `make verify` runs the logic checks;
`make eval report` regenerates every number in §3–4 from the committed cache.
