"""Runnable checks for the pure logic: metrics, reply constraints, guardrail
regexes, thread-root walk, cache keying. No network, no API key."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent import metrics as M
from hiver_agent.draft import check_reply, clean
from hiver_agent.escalate import ANGER_RE, load_intents_escalating
from hiver_agent.text import find_root


def test_escalation_metrics():
    y_true = [True, True, False, False, False]
    y_pred = [True, False, True, False, False]
    m = M.escalation_metrics(y_true, y_pred)
    assert abs(m["escalate_precision"] - 0.5) < 1e-9
    assert abs(m["escalate_recall"] - 0.5) < 1e-9
    # auto-handled = 3 predictions, of which 2 truly safe -> 2/3
    assert abs(m["auto_handle_safety"] - 2 / 3) < 1e-9


def test_bootstrap_ci_brackets_mean():
    ci = M.bootstrap_ci([0, 1, 1, 1, 0, 1], n=500)
    mean = 4 / 6
    assert ci[0] <= mean <= ci[1] + 1e-9 and ci[1] - ci[0] < 1.0


def test_kappa_perfect():
    assert M.kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_reply_constraints():
    assert "over_length" in check_reply("x" * 300)
    assert "link" in check_reply("see http://spoti.fi/help")
    assert "mention" in check_reply("@user check DM")
    assert "pii" in check_reply("email me at a@b.com")
    assert check_reply("We're on it — mind sending us a quick DM?") == []


def test_clean_truncates_and_strips():
    assert "http" not in clean("fix it pls http://x.co " + "y" * 300)
    assert len(clean("word " * 100)) <= 280
    assert "@" not in clean("hey @user here")


def test_anger_regex_and_guardrail_intents():
    assert ANGER_RE.search("this is UNACCEPTABLE, I'm calling my lawyer")
    assert ANGER_RE.search("you charged me twice, I want a chargeback")
    assert not ANGER_RE.search("my playlist is missing songs")
    assert "account_security" in load_intents_escalating()


def test_find_root_walks_parents():
    parent = {2: 1.0, 3: 2.0, 5: 4.0}
    root_of = {}
    assert find_root(3, parent, root_of) == 1  # 3 -> 2 -> 1, 1 has no parent
    assert root_of[2] == 1 and root_of[3] == 1  # memoized chain


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} checks passed")


if __name__ == "__main__":
    main()
