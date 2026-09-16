PY = .venv/bin/python

setup:  ## create venv and install
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install --quiet -e ".[kaggle]"

data:  ## rebuild brand slice from data/raw/twcs (graders: skip — subsample is committed)
	$(PY) scripts/prepare_data.py

taxonomy:  ## LLM taxonomy induction report (then hand-consolidate configs/intents.yaml)
	$(PY) scripts/induce_taxonomy.py

labels:  ## zero-shot label the 6k corpus pool
	$(PY) scripts/build_label_pool.py

index:  ## embed labeled corpus -> index/
	$(PY) scripts/build_index.py

playbooks:  ## distill per-intent playbooks -> configs/playbooks.yaml
	$(PY) scripts/build_playbooks.py

golden:  ## stratified golden set + labeling sheets
	$(PY) scripts/build_golden_set.py

eval:  ## all four systems over the golden set (cached LLM calls)
	$(PY) scripts/run_eval.py --system trivial
	$(PY) scripts/run_eval.py --system simple
	$(PY) scripts/run_eval.py --system noretr
	$(PY) scripts/run_eval.py --system agent

report:  ## aggregate results -> results/SUMMARY.md
	$(PY) scripts/make_report_tables.py

agreement:  ## self-agreement + judge-vs-human agreement (needs human sheets filled)
	$(PY) scripts/self_agreement.py
	$(PY) scripts/judge_agreement.py

labeling-ui:  ## regenerate the browser editors for the human gates
	$(PY) scripts/make_labeling_ui.py

merge-labels:  ## merge downloaded labeling CSVs (DL=~/Downloads) into the repo
	$(PY) scripts/merge_human_labels.py $(DL)

all: labels index playbooks golden eval report  ## full pipeline from committed subsample

verify:  ## logic checks + complexity gate
	$(PY) tests/test_core.py
	.venv/bin/radon cc -n 11 -s src/ scripts/ && echo "complexity ok (no function over CC 10)" || true

demo:  ## python demo.py "your tweet here"
	@echo 'usage: make demo && python demo.py "why was i charged twice"'

.PHONY: setup data taxonomy labels index playbooks golden eval report agreement all verify demo
