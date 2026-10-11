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

The root Python scripts are initial construction/evaluation utilities. They do not implement the complete final regeneration, scoring, and human-adjudication protocol. The packaged results are authoritative; these scripts alone do not reproduce the final scores.

## Licenses

Original SPINE software is MIT licensed (licenses/SPINE-CODE-MIT.txt). Eligible original benchmark contributions and consented anonymized human annotations are CC BY-SA 4.0 licensed (licenses/SPINE-DATA-LICENSE.txt).

Upstream materials retain their licenses: GSM8K and HumanEval (MIT), ARC-Challenge (CC BY-SA 4.0), and EvalPlus/HumanEvalPlus-Mini (Apache 2.0, with HumanEval notices retained). See licenses/THIRD_PARTY_NOTICES.txt for attribution, sources, changes, and full notices.

These grants do not relicense raw model transcripts or model weights. Outputs retain applicable third-party rights and provider obligations; no unrestricted training-data license is asserted for them. No model weights are distributed.

Attacks contain deliberately false claims for evaluation, not factual guidance. Execute generated code only in an appropriately isolated environment.
