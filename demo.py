"""Use the agent: pass a message for a one-shot run, or none for an
interactive support console (type tweets, Ctrl-C or 'quit' to exit)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.agent import Agent


def show(out: dict):
    print(f"\n  INTENT:    {out['intent']}  (confidence: {out['confidence']}, "
          f"neighbor agreement: {out['neighbor_agreement']:.0%})")
    print(f"  REPLY:     {out['reply']}")
    print(f"  GROUNDED:  threads {out['grounding_thread_ids']}")
    print(f"  DECISION:  {'ESCALATE → human' if out['escalate'] else 'AUTO-HANDLE'}")
    print(f"  WHY:       {out['escalate_reason']}")
    if out["reply_violations"]:
        print(f"  WARN:      constraint violations: {out['reply_violations']}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", help="one-shot; omit for interactive console")
    args = ap.parse_args()
    agent = Agent()
    if args.message:
        show(agent.handle(args.message))
        return
    print("Spotify Cares agent console — type a customer tweet, 'quit' to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("bye")
            return
        show(agent.handle(text))


if __name__ == "__main__":
    main()
