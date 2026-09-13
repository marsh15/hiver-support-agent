"""Text + thread helpers shared by the data prep script and tests."""
import re

DM_RE = re.compile(r"\b(dms?|direct messages?)\b", re.I)


def strip_mention(t: str) -> str:
    return re.sub(r"^@\w+\s+", "", str(t)).strip()


def find_root(tid: int, parent: dict, root_of: dict) -> int:
    """Walk the in_response_to chain up to the thread root; memoized."""
    chain = []
    cur = tid
    while cur not in root_of and cur in parent:
        chain.append(cur)
        cur = int(parent[cur])
    r = root_of.get(cur, cur)
    for t in chain:
        root_of[t] = r
    return r
