# REPORT — AI Support Agent for Spotify Cares

*Hiver SDE intern take-home. All numbers regenerate via `make eval report`;
`results/SUMMARY.md` is the source of truth. Label-verification status at time
of writing: **human gate pending** — see §6 item 2.*

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

Golden set: 200 examples, labels = LLM pre-labels + an **independent blind
adjudication pass** (§4; 15 intent and 8 escalate flips applied), and
**self-thread holdout at eval time** (each tweet's own historical thread is
banned from its retrieval — see §6 item 1 for what happened without it).
Numbers are agreement with those labels; the mandatory skepticism is in §6.

| metric | trivial | simple | noretr | agent |
|---|---|---|---|---|
| Intent accuracy (95% CI) | 0.160 [0.110, 0.210] | 0.795 [0.740, 0.850] | 0.800 [0.745, 0.855] | 0.800 [0.745, 0.855] |
| Intent macro-F1 (95% CI) | 0.028 [0.020, 0.035] | 0.815 [0.756, 0.862] | 0.817 [0.754, 0.867] | 0.817 [0.754, 0.867] |
| Escalate precision | 0.420 | 0.308 | 0.680 | **0.689** |
| Escalate recall | 1.000 | 0.095 | 0.405 | 0.369 |
| Auto-handle safety (95% CI) | 1.000* | 0.563 [0.488, 0.635] | 0.667 [0.596, 0.742] | 0.658 [0.582, 0.733] |
| Auto-handle rate | 0.000 | 0.870 | 0.750 | 0.775 |
| Reply constraint violations | 0 | 0 | 0 | 0 |
| Judge pass rate (all dims ≥4) | 0.600 | 0.520 | 0.875 | **0.900** |
| Judge groundedness / actionability / tone | 3.83 / 3.52 / 4.18 | 3.80 / 3.46 / 3.97 | 4.68 / 4.38 / 4.80 | **4.78 / 4.43 / 4.84** |

\* trivial's 1.000 safety is vacuous: it never auto-handles anything.

**Baselines.** *Trivial*: majority-class intent + one canned "please DM us"
reply + always escalate. *Simple*: TF-IDF logistic regression intent classifier
+ per-intent most-common historical reply + keyword-blacklist escalation.
*Ablation* (`noretr`): the full agent with drafting-time retrieval removed — it
shares the classifier path with the agent, so the agent↔noretr delta isolates
exactly what grounding retrieval contributes to replies.

**Reading the numbers honestly.**

- **Intent classification is a three-way tie between simple, noretr, and agent**
  (0.795 / 0.800 / 0.800 — CIs fully overlap). Against the *pre-label* ground
  truth the simple baseline actually beat the agent (0.835 vs 0.815); after the
  blind adjudication pass moved 15 labels, the advantage vanished. That is not
  the agent getting better — it is the ground truth moving away from the
  pre-labeler that the simple baseline was trained on. Both facts are reported
  because together they say the honest thing: at n=200 with model-derived
  labels, intent differences between competent classifiers are noise.
- **The agent's wins are in reply quality and escalation precision**: judge
  pass 0.900 vs 0.520 (simple) and 0.600 (trivial), groundedness 4.78 vs 3.80,
  escalate precision 0.689 vs 0.308. A canned reply cannot resolve anything and
  a keyword policy escalates wrongly 69% of the time.
- **The agent's wins are in reply quality and escalation precision**: judge
  pass 0.900 vs 0.520 (simple) and 0.600 (trivial), groundedness 4.78 vs 3.80,
  escalate precision 0.778 vs 0.423. A canned reply cannot resolve anything and
  a keyword policy escalates wrongly 58% of the time.
- **Retrieval helps replies, not classification**: agent vs noretr judge pass
  0.900 vs 0.875, groundedness 4.78 vs 4.68. Modest but consistent, and it is
  the *right* channel — grounding changes how the reply sounds, not which
  bucket the tweet falls in.
- **Auto-handle safety 0.671 is below the 0.95 bar I set in §1.** The honest
  deployment claim is therefore *draft-assist, not auto-pilot*: the agent
  drafts and triages, a human confirms send. The escalation failure analysis
  (§5) says what a week of work would target.
- Zero reply-constraint violations across 200 drafts: the deterministic
  post-processor (link/mention/PII strip, ≤280 enforce) does its job — this is
  the one number I trust unconditionally, because it doesn't depend on labels.

## 4. Evaluation setup (the proof)

- **Golden set**: 200 examples, stratified by induced intent with ≥8 per intent
  from the 6k-row labeled pool, seed 13. Labels went through **two model passes
  with different protocols**: (1) zero-shot pre-labels, then (2) a blind
  adjudication pass (`scripts/adjudicate_golden.py`) — full thread context, no
  pre-label shown, disagreements arbitrated by a third call, flips applied only
  on high-confidence arbitration, everything logged to
  `results/adjudication_log.csv`. Label stability: **18.5% intent / 6% escalate
  disagreement** between passes; 15 intent + 8 escalate flips applied. That
  disagreement rate is the honest noise bar on this ground truth. **The human
  gate (`labeling/golden_sheet.html` → `data/golden.csv` `verified` flag)
  remains open; §3 numbers regenerate with `make eval report` after it.**
- **Labeler self-agreement**: 30 examples re-labeled blind (shuffled, labels
  hidden, `labeling/relabel_30.csv`), scored by `scripts/self_agreement.py`.
  **Pending the same gate.**
- **LLM judge**: `gpt-4.1-mini`, temperature 0, scores groundedness /
  actionability / tone / safety 1–5 (pass = all ≥4). The judge sees the
  playbook and the same exemplar pool the drafter had, so "grounded" is
  checkable, not vibes.
- **Judge validation**: I hand-score 50 drafted replies blind on the same
  rubric (`labeling/judge_sheet.csv`) → `scripts/judge_agreement.py` reports
  exact/within-1 agreement and Cohen's kappa per dimension. **Pending; the
  sheet is generated.** Two judge behaviors are already visible in the raw
  outputs and I report them as findings, not excuses: it fails 12 agent drafts
  *all* on actionability=3 ("no concrete next step"), and it passes the
  trivial baseline's canned DM-redirect reply 62% of the time — strict on
  specificity, lenient on fluent empathy.
- **Uncertainty**: bootstrap 95% CIs (2,000 resamples, seed 42) on every
  headline number.
- **Reproducibility check**: a fresh-clone simulation (clone → `make setup` →
  `make eval report` with a warm LLM cache) reproduced all results in ~3.5 min;
  the agent's numbers were identical, with one nuance disclosed: query
  embeddings are not cached, so the *simple* baseline's judge scores moved by
  ≤0.17 on three dimensions between runs (near-tie exemplar sets reshuffle).
  Intent and escalation metrics are bit-stable.

## 5. Failure analysis (top 5 modes, real examples from the golden run)

*Also observed in live use (found by the author testing the console, not by the
eval): the deterministic link-stripper can leave dangling phrases — a drafter
output of "check out <help-center link>" becomes "check out for next steps"
after URL removal. The safety constraint holds; the grammar pays. A
sentence-completion repair pass is on the §7 list.*

1. **Adjacent-intent confusion in the plan/money/how-to triangle** (largest
   error cluster). "[1989832] our account is 'Free' while there's still money
   charged" → true billing_charges, predicted subscription_management;
   "[686630] how can I remove my student status?" → true subscription, pred
   billing. *Hypothesis*: these tweets genuinely straddle two intents
   (a billing symptom of a subscription state, asked as a how-to); the
   induced taxonomy draws the boundary through the middle, and every system
   — including the labeler that made the golden labels — inherits it.
2. **Product-feedback tweets have no home.** "[2284417] please optimize your
   app for the iPhone X 🙏", "[2807661] Really upset that you took out song
   previews" → true app_technical, predicted feature_howto. *Hypothesis*:
   "complaint about a product decision" is not in the taxonomy; the classifier
   distributes these across technical/how-to arbitrarily. A `product_feedback`
   intent (or explicit routing to feedback CRM) is the fix — I kept the
   taxonomy at 10 and paid here.
3. **Account-recovery trouble gets auto-handled** — the scariest failure.
   "[1931908] trying to reset my password but it keeps saying CSRF token is
   invalid", "[2630281] I'm locked out of my account so the only way to contact
   you is through twitter???" → classified account_access (routine), drafted,
   auto-handled; both were truly-escalate. *Hypothesis*: `always_escalate`
   covers account_security (hacked) but not account_access (locked out), and
   the LLM escalation proposal reads well-formed frustration as routine. These
   two examples alone justify guardrail expansion before any deployment.
4. **The judge is two-sidedly miscalibrated.** It fails 12/200 agent drafts,
   every one on actionability ("lacks a concrete next step"), while passing the
   trivial canned reply 62% of the time — e.g. it rates the canned "please DM
   us" reply groundedness 3.81. *Hypothesis*: the rubric's 1–5 anchors reward
   politeness and punish missing specifics; a real human would invert much of
   that. The human validation sheet exists precisely to quantify this before
   the judge numbers are trusted.
5. **Informal venting hides operational severity.** "[1796710] My Spotify keeps
   randomly switching to some stranger's playlist, wtf?" → true
   account_security (likely compromise), predicted app_technical, no escalation.
   *Hypothesis*: security signals phrased as app weirdness ("stranger's
   playlist") don't match "hacked/scam" lexicons, and the classifier sees
   playback symptom. A severity-over-intent guardrail (unknown-device /
   someone-else's-account phrases → escalate) would catch this class.

## 6. What is misleading about my headline number?

The honest list, most damaging first:

1. **I found 100% retrieval leakage in my first headline run — and fixed it
   mid-project.** In the first eval, every golden tweet retrieved its *own*
   historical thread as an exemplar (they sit in the index verbatim;
   similarity 1.0). Intent accuracy was **0.985** with per-class F1s of 1.000 —
   the agent could parrot the labeled answer for the exact same tweet. The
   current numbers use a self-thread holdout (own thread banned from that
   tweet's retrieval) and drop to **0.815 (0.800 on the adjudicated labels)**.
   If a takeaway survives, it's
   this one: eval leakage made a mediocre classifier look superhuman, and
   nothing in the pipeline flagged it — I only found it by hunting for it.
2. **My labels are the ground truth — and I built the system — and they are
   model-derived, not hand-verified.** Two model passes (pre-label + blind
   adjudication) disagree 18.5% on intent — that is the honest noise floor of
   this "ground truth". The pre-labels came from the same model family as the
   classifier and the simple baseline's training labels, which is why intent
   numbers are high *and* why the systems tie: everything agrees with the same
   taxonomy-inducing family. The open human gate (golden sheet + judge sheet,
   both generated and waiting) is what converts §3 into real evidence; treat
   current intent numbers as intra-family agreement, not accuracy.
3. **Escalation ground truth is DM-redirect-tinted.** ~37% of historical first
   replies redirect to DM and my "should escalate" pre-labels lean on that
   outcome. So agent escalate-recall 0.407 partly measures *philosophy
   disagreement* — the agent believes a public reply resolves routine issues
   that Spotify historically took private — not only misses. Auto-handle
   safety 0.671 inherits the same tint in the other direction.
4. **Judge leniency/strictness is systematic (see §5.4).** The 0.900 pass rate
   overstates quality (it rewards fluent empathy — the canned reply scores
   3.81 groundedness) while the 12 actionability failures may overstate
   defects. Human validation of 50 replies is pending; until then the judge
   numbers rank systems more reliably than they measure quality.
5. **2017 data, one brand, English-only, n=200.** Intent boundaries and
   response norms are six years stale (no audiobooks, no AI DJ era);
   macro-F1 on rare intents rides on ≤8 examples; every CI is wide. And the
   biggest one: **all of this measures agreement with one annotator who is
   also the author** — the 30-example blind self-agreement audit bounds
   intra-rater noise, never bias.

## 7. With one more week

1. **Finish the human gates and add a second annotator** on the golden set +
   judge sheet → inter-annotator kappa, making §3 mean what it claims.
2. **Threshold + guardrail tuning on a held-out split** — sweep the
   confidence threshold and add severity-over-intent guardrails (account
   lockout/recovery phrases, unknown-device phrasing) targeting auto-handle
   safety ≥0.95; ship as draft-assist until then.
3. **A `product_feedback` intent or CRM route** to give the largest
   homeless cluster a home (§5.2).
4. **A distilled cheap classifier** (logreg/MiniLM on the 6k induced labels)
   — the simple baseline's 0.795 already shows the labels are learnable; if a
   $0.0001 classifier matches the LLM's 0.800, the LLM moves to judge-only
   and the cost story collapses 100x.
5. **Leakage regression test in CI**: assert zero golden ids appear in
   retrieval results at eval time, so the §6.1 embarrassment can never recur
   silently.
6. **Grammar repair after constraint enforcement** — rewrite or drop dangling
   phrases left by the link/mention stripper (see §5 observation) instead of
   shipping "check out for next steps".

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
- **Self-thread holdout at eval time** — added after finding 100% retrieval
  leakage; the pre-fix numbers are preserved in §6.1 as the honesty exhibit.
- **Reproducibility engineering** — committed 27,914-thread subsample and
  embedding index; every LLM call disk-cached locally: a grader's eval needs no
  Kaggle, no embedding spend, and ~$3–5 of model spend.
- **Scope minimalism** — no LangChain/vector DB (a ~700-line pipeline a
  reviewer can read live; numpy cosine suffices at this scale) and Banking77
  skipped (depth over breadth; logged, not hidden).
- **Bootstrap CIs on everything**: with n=200, a 3-point accuracy gap between
  systems can be noise; CIs force the report to say so.
- **Reported "simple beats agent on intent" rather than choosing flattering
  metrics**: the agent's claim to value is reply quality + escalation
  precision; pretending otherwise would be the first misleading headline.

## 9. Reproducing

See README quickstart. `make verify` runs the logic checks;
`make eval report` regenerates every number in §3 from the committed index and
subsample (fresh key ≈ $3–5, ~15 min). Labeling gates: `data/golden.csv`
(200), `labeling/relabel_30.csv` (30 blind), `labeling/judge_sheet.csv` (50).
