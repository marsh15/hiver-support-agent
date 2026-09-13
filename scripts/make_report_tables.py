"""Aggregate results/<system>/metrics.json into results/SUMMARY.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.config import ROOT

ROWS = [
    ("Intent accuracy (95% CI)", lambda m: f"{m['intent']['accuracy']:.3f} "
     f"[{m['intent']['accuracy_ci95'][0]:.3f}, {m['intent']['accuracy_ci95'][1]:.3f}]"),
    ("Intent macro-F1 (95% CI)", lambda m: f"{m['intent']['macro_f1']:.3f} "
     f"[{m['intent']['macro_f1_ci95'][0]:.3f}, {m['intent']['macro_f1_ci95'][1]:.3f}]"),
    ("Escalate precision", lambda m: f"{m['escalation']['escalate_precision']:.3f}"),
    ("Escalate recall", lambda m: f"{m['escalation']['escalate_recall']:.3f}"),
    ("Escalate F1", lambda m: f"{m['escalation']['escalate_f1']:.3f}"),
    ("Auto-handle safety (95% CI)", lambda m: f"{m['escalation']['auto_handle_safety']:.3f} "
     f"[{m['escalation']['auto_handle_safety_ci95'][0]:.3f}, {m['escalation']['auto_handle_safety_ci95'][1]:.3f}]"),
    ("Auto-handle rate", lambda m: f"{m['escalation']['auto_handle_rate']:.3f}"),
    ("Reply constraint violations", lambda m: f"{m['reply_constraint_violation_rate']:.3f}"),
    ("Judge: groundedness", lambda m: f"{m['judge']['mean_groundedness']:.2f}"),
    ("Judge: actionability", lambda m: f"{m['judge']['mean_actionability']:.2f}"),
    ("Judge: tone", lambda m: f"{m['judge']['mean_tone']:.2f}"),
    ("Judge: safety", lambda m: f"{m['judge']['mean_safety']:.2f}"),
    ("Judge: pass rate (all >=4)", lambda m: f"{m['judge']['pass_rate']:.3f}"),
]


def main():
    systems, ms = [], []
    for s in ["trivial", "simple", "noretr", "agent"]:
        p = ROOT / "results" / s / "metrics.json"
        if p.exists():
            systems.append(s)
            ms.append(json.loads(p.read_text()))
    if not ms:
        raise SystemExit("no results yet — run `make eval` first")

    lines = ["# Results summary", "",
             f"Golden set verified by human: {all(m['golden_verified'] for m in ms)}", "",
             "| metric | " + " | ".join(systems) + " |",
             "|---" * (len(systems) + 1) + "|"]
    for label, fn in ROWS:
        lines.append(f"| {label} | " + " | ".join(fn(m) for m in ms) + " |")
    lines += ["", "## Per-class intent F1 (agent)", ""]
    agent = next(m for m in ms if m["system"] == "agent")
    for k, v in sorted(agent["intent"]["per_class_f1"].items(), key=lambda kv: kv[1]):
        lines.append(f"- {k}: {v:.3f}")
    (ROOT / "results/SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
