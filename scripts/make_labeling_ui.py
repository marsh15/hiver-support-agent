"""Generate self-contained browser editors for the two human gates:
  labeling/golden_editor.html  — verify/correct 200 golden labels
  labeling/judge_editor.html   — blind-score 50 replies on the rubric
Each embeds its data, keeps progress in localStorage, and exports a CSV in the
exact column format of the file it replaces. No server, works offline."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.classify import load_intents
from hiver_agent.config import ROOT

CSS = """body{font-family:-apple-system,Segoe UI,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#fafafa;color:#111}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px 20px;margin:14px 0}
.tweet{font-size:16px;line-height:1.45}
.thread{background:#f4f6f8;border-radius:8px;padding:10px;font-size:13px;white-space:pre-wrap;margin-top:8px}
.reply{background:#eef7ee;border-radius:8px;padding:10px;font-size:15px;margin-top:8px}
.row{margin-top:12px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
button{padding:7px 14px;border-radius:8px;border:1px solid #bbb;background:#fff;cursor:pointer;font-size:14px}
button.primary{background:#1db954;color:#fff;border-color:#1db954;font-weight:600}
button.on{outline:2px solid #1db954}
select{padding:6px;font-size:14px}
.nav{position:sticky;top:0;background:#fafafa;padding:10px 0;display:flex;gap:12px;align-items:center;z-index:5}
.prog{font-weight:600}
h3{margin:4px 0}"""

JS_COMMON = """
const DATA = __DATA__;
const KEY = "__KEY__";
const COLS = __COLS__;
const OUT = "__OUT__";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = parseInt(localStorage.getItem(KEY + "_i") || "0");
function save(){localStorage.setItem(KEY, JSON.stringify(state));localStorage.setItem(KEY+"_i", i);}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function complete(s){ return __COMPLETE_TEST__; }
function dl(){
  const rows = DATA.map(r => Object.assign({}, r, state[r.id] || {}));
  const csv = [COLS.join(",")].concat(rows.map(r => COLS.map(c => {
    let v = r[c] === undefined ? "" : r[c]; v = String(v);
    return /[",\\n]/.test(v) ? '"' + v.replace(/"/g,'""') + '"' : v;
  }).join(","))).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = OUT; a.click();
}
function upd(k, v){
  const r = DATA[i];
  state[r.id] = state[r.id] || {};
  state[r.id][k] = v;
  state[r.id].__complete__ = complete(state[r.id]);
  save();
  if(state[r.id].__complete__) setTimeout(()=>{ i = Math.min(i+1, DATA.length); save(); render(); }, 150);
  render();
}
document.querySelectorAll("button[data-k]").forEach(b =>
  b.addEventListener("click", () => upd(b.dataset.k, b.dataset.v)));
const sel = document.getElementById("intent_sel");
if (sel) sel.addEventListener("change", () => upd("intent", sel.value));
document.getElementById("prev").onclick = () => { i = Math.max(0, i-1); save(); render(); };
document.getElementById("next").onclick = () => { i = Math.min(DATA.length-1, i+1); save(); render(); };
render();
"""

JS_RENDER_INTENT = """
function render(){
  const done = DATA.filter(r => state[r.id] && state[r.id].__complete__).length;
  document.getElementById("prog").textContent = done + " / " + DATA.length + " done";
  const r = DATA[i];
  if(!r){ document.getElementById("card").innerHTML = "<h2>All done — click Download CSV.</h2>"; return; }
  const s = state[r.id] || {};
  document.getElementById("tweet").textContent = r.text;
  document.getElementById("thread").textContent = r.thread;
  const sel = document.getElementById("intent_sel");
  sel.value = s.intent || r.pre_intent;
  document.getElementById("esc_true").className = s.escalate === "true" ? "on" : "";
  document.getElementById("esc_false").className = s.escalate === "false" ? "on" : "";
  document.getElementById("pre").textContent = r.pre_intent !== undefined
    ? "pre-label: " + r.pre_intent + " / " + r.pre_escalate : "(blind — no pre-label shown)";
}
"""

JS_RENDER_JUDGE = """
function render(){
  const done = DATA.filter(r => state[r.id] && state[r.id].__complete__).length;
  document.getElementById("prog").textContent = done + " / " + DATA.length + " done";
  const r = DATA[i];
  if(!r){ document.getElementById("card").innerHTML = "<h2>All done — click Download CSV.</h2>"; return; }
  const s = state[r.id] || {};
  document.getElementById("tweet").textContent = r.text;
  document.getElementById("reply").textContent = r.reply;
  ["groundedness","actionability","tone","safety"].forEach(d =>
    [1,2,3,4,5].forEach(n => {
      const b = document.getElementById(d + "_" + n);
      b.className = s[d] === String(n) ? "on" : "";
    }));
  document.getElementById("pass_true").className = s.human_pass === "true" ? "on" : "";
  document.getElementById("pass_false").className = s.human_pass === "false" ? "on" : "";
}
"""

GOLDEN_BODY = """
<div class="nav"><button class="primary" id="dl">Download CSV</button>
<span class="prog" id="prog"></span><span style="color:#666">correct intent + escalate where wrong; auto-advances when both are set</span></div>
<div class="card" id="card">
<div class="tweet" id="tweet"></div>
<details><summary>real thread</summary><div class="thread" id="thread"></div></details>
<div class="row"><b>intent:</b>
<select id="intent_sel">__INTENT_OPTIONS__</select>
<b>escalate:</b>
<button id="esc_true" data-k="escalate" data-v="true">escalate</button>
<button id="esc_false" data-k="escalate" data-v="false">auto</button>
<span id="pre" style="color:#888"></span></div>
<div class="row"><button id="prev">&larr; prev</button><button id="next">next &rarr;</button></div>
</div>"""

JUDGE_BODY = """
<div class="nav"><button class="primary" id="dl">Download CSV</button>
<span class="prog" id="prog"></span><span style="color:#666">score 1–5 before forming an opinion of the agent; 5 = exactly what Spotify would say</span></div>
<div class="card" id="card">
<div><b>customer:</b> <span class="tweet" id="tweet"></span></div>
<div class="reply"><b>agent reply:</b> <span id="reply"></span></div>
<div class="row">
<span><b>grnd:</b>__SCALE__g</span>
<span><b>actn:</b>__SCALE__a</span>
<span><b>tone:</b>__SCALE__t</span>
<span><b>safe:</b>__SCALE__s</span></div>
<div class="row"><b>pass:</b>
<button id="pass_true" data-k="human_pass" data-v="true">pass</button>
<button id="pass_false" data-k="human_pass" data-v="false">fail</button></div>
<div class="row"><button id="prev">&larr; prev</button><button id="next">skip &rarr;</button></div>
</div>"""


def scale_buttons(dim: str) -> str:
    return "".join(
        f'<button id="{dim}_{n}" data-k="{dim}" data-v="{n}">{n}</button>'
        for n in (1, 2, 3, 4, 5))


def page(title: str, body: str, js_render: str, data: list, key: str,
         out: str, cols: list, complete_test: str) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
{body}
<script>
const DATA = {json.dumps(data)};
const KEY = "{key}";
const COLS = {json.dumps(cols)};
const OUT = "{out}";
let state = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let i = parseInt(localStorage.getItem(KEY + "_i") || "0");
function save(){{localStorage.setItem(KEY, JSON.stringify(state));localStorage.setItem(KEY+"_i", i);}}
function esc(s){{return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")}}
function complete(s){{ return {complete_test}; }}
function dl(){{
  const rows = DATA.map(r => Object.assign({{}}, r, state[r.id] || {{}}));
  const csv = [COLS.join(",")].concat(rows.map(r => COLS.map(c => {{
    let v = r[c] === undefined ? "" : r[c]; v = String(v);
    return /[",\\n]/.test(v) ? '"' + v.replace(/"/g,'""') + '"' : v;
  }}).join(","))).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type:"text/csv"}}));
  a.download = OUT; a.click();
}}
document.getElementById("dl").onclick = dl;
function upd(k, v){{
  const r = DATA[i];
  state[r.id] = state[r.id] || {{}};
  state[r.id][k] = v;
  state[r.id].__complete__ = complete(state[r.id]);
  save();
  if(state[r.id].__complete__) setTimeout(()=>{{ i = Math.min(i+1, DATA.length); save(); render(); }}, 150);
  render();
}}
document.querySelectorAll("button[data-k]").forEach(b =>
  b.addEventListener("click", () => upd(b.dataset.k, b.dataset.v)));
const sel = document.getElementById("intent_sel");
if (sel) sel.addEventListener("change", () => upd("intent", sel.value));
document.getElementById("prev").onclick = () => {{ i = Math.max(0, i-1); save(); render(); }};
document.getElementById("next").onclick = () => {{ i = Math.min(DATA.length-1, i+1); save(); render(); }};
{js_render}
</script></body></html>"""


def build_golden():
    g = pd.read_csv(ROOT / "data/golden.csv")
    intents = list(load_intents())
    data = [{"id": int(r.root_tweet_id), "text": r.text, "thread": r.thread_head,
             "pre_intent": r.pre_intent, "pre_escalate": bool(r.pre_escalate)}
            for r in g.itertuples()]
    opts = "".join(f'<option value="{n}">{n}</option>' for n in intents)
    body = GOLDEN_BODY.replace("__INTENT_OPTIONS__", opts)
    html = page("Golden set verification (200)", body, JS_RENDER_INTENT, data,
                "golden_gate", "golden_corrected.csv", ["id", "intent", "escalate"],
                "(s.intent !== undefined && s.escalate !== undefined)")
    (ROOT / "labeling/golden_editor.html").write_text(html)
    return len(data)


def build_judge():
    js = pd.read_csv(ROOT / "labeling/judge_sheet.csv")
    data = [{"id": int(r.root_tweet_id), "text": r.text, "reply": r.reply}
            for r in js.itertuples()]
    body = (JUDGE_BODY
            .replace("__SCALE__g", scale_buttons("groundedness"))
            .replace("__SCALE__a", scale_buttons("actionability"))
            .replace("__SCALE__t", scale_buttons("tone"))
            .replace("__SCALE__s", scale_buttons("safety")))
    html = page("Judge validation — blind human scoring (50)", body, JS_RENDER_JUDGE,
                data, "judge_gate", "judge_sheet_filled.csv",
                ["id", "groundedness", "actionability", "tone", "safety", "human_pass"],
                "['groundedness','actionability','tone','safety','human_pass'].every(k => s[k] !== undefined)")
    (ROOT / "labeling/judge_editor.html").write_text(html)
    return len(data)


def build_relabel():
    """Blind self-agreement editor: no pre-labels shown, no golden peeking."""
    r30 = pd.read_csv(ROOT / "labeling/relabel_30.csv")
    r30 = r30.head(30)
    intents = list(load_intents())
    data = [{"id": int(r.root_tweet_id), "text": r.text, "thread": r.thread_head}
            for r in r30.itertuples()]
    opts = "".join(f'<option value="{n}">{n}</option>' for n in intents)
    body = GOLDEN_BODY.replace("__INTENT_OPTIONS__", opts)
    html = page("Blind relabel (30) — self-agreement audit", body, JS_RENDER_INTENT,
                data, "relabel_gate", "relabel_30_filled.csv",
                ["id", "intent", "escalate"],
                "(s.intent !== undefined && s.escalate !== undefined)")
    (ROOT / "labeling/relabel_editor.html").write_text(html)
    return len(data)


if __name__ == "__main__":
    print(f"labeling/golden_editor.html: {build_golden()} rows")
    print(f"labeling/judge_editor.html: {build_judge()} rows")
    print(f"labeling/relabel_editor.html: {build_relabel()} rows (blind)")
    print("open in a browser; label; Download CSV; follow README gate instructions")
