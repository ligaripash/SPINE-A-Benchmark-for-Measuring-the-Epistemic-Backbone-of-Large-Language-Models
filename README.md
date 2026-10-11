# SPINE: A Benchmark for Measuring the Epistemic Backbone of Large Language Models

SPINE pairs resistance to false persona-framed challenges with appropriate updating after explicit task changes in mathematics, code, and science.

## Data and results

- data/spine_dataset_persona_validated_v3.json: 753 seeds, 4,518 attacks, and 753 corrections. code_39 is quarantined from follow-up scoring.
- results/final_response_index.jsonl.gz: 29,221 source-linked initial and follow-up responses, with task, conversation context, model identifiers, and provenance hashes. Read using Python gzip.open with UTF-8 text mode.
- results/final_correctness_labels.json: authoritative final labels keyed by blind_id; null means unresolved. Use these rather than the index's earlier primary_correct field.
- results/human_overrides.json: 14 anonymized human label corrections. Private reviewer exports and identities are excluded.
- results/metrics.json and results/RESULTS.md: final scores, coverage, and uncertainty. ESS measures retention; AUR measures updating; BES is their harmonic mean.
- results/final_verification.json: verified source hashes and bootstrap settings. Decompress the response file before comparing its source hash.
- figures/figure1_arr.pdf: current paper figure.
- RELEASE_MANIFEST.json: checksums and sizes of the packaged artifacts, including the compressed response file.

## Reproduce the reported results (offline)

Use Python 3.12. The commands below require no model servers, API keys, or paid calls. Run them from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce/verify_release.py
python reproduce/recompute_metrics.py
python reproduce/build_figure.py
```

`recompute_metrics.py` starts from the released response index and final labels, reconstructs model-specific and shared eligibility cohorts, and recomputes all 168 model/domain/persona metric rows in each cohort. It checks accuracy, ESS, AUR, BES, Overall, correct/resolved counts, unresolved-label bounds, and both 10,000-replicate domain-stratified whole-seed bootstrap analyses against `results/metrics.json`. It uses the published sampling seed (20261004) and paired draws across models. Numerical comparisons use relative tolerance 1e-10 and absolute tolerance 1e-12. For a quick check, add `--skip-bootstrap`; that does not verify confidence intervals.

Outputs go to `reproduced/`; published files are not overwritten. The figure renderer writes `reproduced/figures/figure1_arr.pdf` and its numerical provenance. It uses portable standard PDF fonts, so its typography and PDF bytes can differ from the archived figure while using identical unrounded persona means. The offline commands were verified with Python 3.12.14, NumPy 2.3.5, and ReportLab 4.4.9. These are the reproduction environment versions, not a claim about every historical experiment environment.

This reproduces numerical results conditional on the released responses and labels. It does not regenerate model outputs or independently validate the labels. Mutable API aliases, sampling, and infrastructure differences prevent a guarantee that fresh inference will recreate recorded text.

## Software included

`software/` contains the construction, validation, generation, recovery, scoring, adjudication, human-review, and analysis source files. See [the software guide](software/README.md) for the pipeline map and checkpoint requirements. `software/SOURCE_MANIFEST.json` records source hashes and disclosed anonymization changes. Private reviewer names in code were replaced by role aliases; raw reviewer exports are not included. `reproduce/metrics_core.py` contains the numerical functions used by the final analysis, extracted without importing runners or making API calls.

The root-level scripts are the initial utilities. The final pipeline sources are under `software/`; do not use the initial root scripts alone to claim reproduction of the final evaluation.

## Build blank human-review questionnaires

```bash
python reproduce/build_review.py --kind science
python reproduce/build_review.py --kind dataset
```

Open `review_workspace/science/primary_review.html` or `review_workspace/dataset/primary_review.html` in a browser. The science questionnaire reconstructs the 100 audited response slots from released stimuli, without human decisions, AI labels, or model identities in the questionnaire. Its saved ID list reproduces the selected sample; it is not a new random sample or a rerun of the original stratified selection. The dataset builder reconstructs the 180-item packet and 54-item secondary packet, not a claim of 180 completed reviews. Reviewers should use pseudonymous IDs, save each item, and export a backup. Keep completed exports private. The review workspace is git-ignored. Dataset references are hidden in the interface but embedded in the HTML; reviewers should not inspect the source before recording independent answers.

## Release checks

The release was checked for known credential patterns, literal credential assignments, private reviewer identifiers, and personal user-directory paths. Python sources were syntax-compiled. Only source files and blank templates were added; browser storage, credentials, review exports, intermediate receipts, and raw reviewer notes were not copied. `SOFTWARE_VERIFICATION.json` records the performed checks. Such scanning is a documented check, not a guarantee against every possible secret format.

## Licenses

Original SPINE software is MIT licensed (licenses/SPINE-CODE-MIT.txt). Eligible original benchmark contributions and consented anonymized human annotations are CC BY-SA 4.0 licensed (licenses/SPINE-DATA-LICENSE.txt).

Upstream materials retain their licenses: GSM8K and HumanEval (MIT), ARC-Challenge (CC BY-SA 4.0), and EvalPlus/HumanEvalPlus-Mini (Apache 2.0, with HumanEval notices retained). See licenses/THIRD_PARTY_NOTICES.txt for attribution, sources, changes, and full notices.

These grants do not relicense raw model transcripts or model weights. Outputs retain applicable third-party rights and provider obligations; no unrestricted training-data license is asserted for them. No model weights are distributed.

Attacks contain deliberately false claims for evaluation, not factual guidance. Execute generated code only in an appropriately isolated environment.
