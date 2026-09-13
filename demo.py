"""Run one message through the full agent — the live-demo entry point."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.agent import Agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    args = ap.parse_args()
    out = Agent().handle(args.message)
    print(f"TWEET:     {out['text']}")
    print(f"INTENT:    {out['intent']} (confidence: {out['confidence']}, "
          f"neighbor agreement: {out['neighbor_agreement']:.0%})")
    print(f"REPLY:     {out['reply']}")
    print(f"           grounded in threads: {out['grounding_thread_ids']}")
    print(f"DECISION:  {'ESCALATE' if out['escalate'] else 'AUTO-HANDLE'} — {out['escalate_reason']}")
    if out["reply_violations"]:
        print(f"WARNING: reply constraint violations: {out['reply_violations']}")


if __name__ == "__main__":
    main()
