# SPINE software guide

This directory preserves the research pipeline's relative source layout. Most scripts are checkpoint-oriented research utilities, not interchangeable command-line entry points. Use `../reproduce/` for the tested, self-contained offline reproduction path.

## Pipeline map

| Stage | Main source files |
| --- | --- |
| Source provenance and reference checks | `original_validation/source_crosswalk.py`, `check_gsm8k_arithmetic.py`, `check_humaneval.py`, `build_corrected_dataset.py` |
| Candidate corrective interventions | `valid_corrections/generate_gpt4o_candidates.py`, `generate_reasoned_candidates.py`, `build_dataset.py` |
| Corrective validation and repair | `valid_corrections/corrective_target_validation_v1/`, `corrective_target_repair_v2/`, `verify_code_corrections.py`, `verify_math_gemini.py` |
| Attack factual and persona checks | `valid_corrections/persona_alignment_review.py`, `repair_false_claim_attacks.py`, `run_persona_attack_fixer.py`, `verify_persona_attack_fixes.py` |
| Executable attack witnesses | `valid_corrections/audit_repaired_code_attacks.py`, `audit_retained_code_attacks.py`, `code_attack_reference_worker.py` |
| Frozen dataset assembly | `model_runs/finalize_persona_dataset_v3.py` |
| Initial and corrective target generation | `model_runs/run_four_models.py`, `llama3_8b/`, `llama3_8b_revised/`, `corrective_branch_v2/reevaluate.py` |
| Changed-attack generation | `model_runs/reevaluate_persona_v3.py`, `resume_persona_scoring_v3.py` |
| Output-budget recovery | `model_runs/audit_output_budget_v4.py`, `recover_local_initial_budget_v4.py`, `recover_api_and_score_initial_v4.py`, `recover_gemini_initial_v4.py`, `run_local_remaining_v4.py`, `resume_api_followups_v4.py`, `extend_deepseek_followups_v4.py` |
| Numerical and functional scoring | `model_runs/llama3_8b/score_objective.py`, `score_code.py`, `model_runs/finalize_output_budget_v4.py` |
| Science extraction and arbitration | `model_runs/judge_robustness/run_judges.py`, `run_sol_adjudication.py`, `run_sol_clarification.py`, `run_sol_agreed_unclear.py` |
| Human review and reconciliation | `valid_corrections/build_dataset_validity_review_v3.py`, `model_runs/build_science_human_review_v3.py`, `build_science_cross_review_v4.py`, `build_science_reconciliation_v4.py`, associated comparison scripts and HTML templates |
| Final human labels and metrics | `model_runs/apply_human_labels_v5.py`, `judge_robustness/analyze_sol_adjudication.py`, `judge_robustness/analyze_judges.py` |
| Additional sensitivity analyses | `model_runs/analyze_persona_publication_v3.py`, `judge_robustness/`, `corrective_branch_v2/` |

Older analysis scripts contain fixed cohort sizes and assertions for their specific stages. Those are not settings for the final release. The tested reproduction wrapper uses the final index and generalized seed count exactly as in the final runner. Paper-editing automation, machine-specific launch scripts, quota probes, and archived duplicate source copies are omitted; they are not needed to compute results.

## Dependencies and inputs

The final pipeline uses Python's standard library plus NumPy; plotting uses ReportLab. The initial root utilities have optional dependencies listed in `../requirements-legacy.txt`. Model generation additionally needs Ollama and/or provider API access to the recorded model identifiers. API availability is not assured.

Historical scripts resolve paths relative to this directory as their workspace root. They expect frozen datasets, source-cache tests, run manifests, request packets, and prior receipts at their specified paths. These intermediate checkpoints are **not all included** in the public release. Copying the final dataset into an older filename is not a valid substitute because source hashes and cohort semantics differ. Thus, the historical resume scripts cannot be run on a fresh checkout to resume the authors' processes. They are provided to disclose algorithms, prompts, parameter settings, and request/retry policies. Reproducing the final numerical tables from released responses does not require these private or intermediate inputs.

Human review scripts that compare exported decisions require new, locally collected review exports. The public release contains consented final label overrides but excludes reviewer identities and private raw exports; it cannot independently reconstruct every earlier reviewer judgment. Blank questionnaire reconstruction is supported by the commands in the root README.

## Running a new generation study

Review the stage-specific source and establish a new workspace with its own manifests and source inputs. Configure the model identities and use the recorded generation parameters if comparing protocols. Keep provider keys in environment variables or masked prompts; never commit them. Historical runners save successful receipts and stop on unknown transport completion. Investigate `.inflight` markers before resuming; do not blindly delete them or repeat requests. Do not run multiple writers against one checkpoint directory. API generation incurs charges and is not part of the offline reproduction commands.

Functional scoring executes model-generated Python. Its subprocess restrictions and timeouts are not a hardened security boundary. Run that stage in an appropriately isolated environment. The offline metric/figure commands do not execute model-generated code.

## Source modifications

Python source files retain their computational logic. Personal reviewer identifiers are replaced by `reviewer_a`, `reviewer_b`, or `dataset_reviewer` in filenames and code. `SOURCE_MANIFEST.json` records original and distributed hashes and these transformations. No private review exports or keys are needed for the offline commands. New portable wrappers are clearly located in `reproduce/`, separate from the historical source.
