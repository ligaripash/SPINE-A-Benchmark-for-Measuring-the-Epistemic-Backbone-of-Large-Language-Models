"""Summarize only completed v2 evaluations; pending models have no new scores."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ['llama3_8b_revised','qwen2_5_7b','mistral_nemo','deepseek_r1_1_5b','gpt_4o_mini','gemini_3_flash']


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def percent(value):
    return f'{100*value:.2f}' if value is not None else '—'


def main():
    completed = []
    progress = []
    table = ['| Model | Accuracy % | ESS % | AUR % | BES % | Overall % | AUR correct / total | Unresolved AUR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    persona = ['| Model | Sophistry | Authority | Consensus | Emotional | Doubter | Bureaucrat |',
               '|---|---:|---:|---:|---:|---:|---:|']
    uncertainty = ['| Model | ESS correct / resolved | Unresolved attacks / all attacks | ESS bounds % | BES bounds % | Overall bounds % |',
                   '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        directory = HERE/name
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        raw = {x['id']:x for x in read_rows(directory/'correction_new.jsonl') if 'error' not in x}
        status = {'run_name':name,'fresh_completed':len(raw),'fresh_required':manifest['expected_new_requests'],
                  'reused_responses':manifest['correction_reused'],'eligible':manifest['correction_eligible'],
                  'generation_active':(directory/'generation.lock').exists(),
                  'scoring_active':(directory/'scoring.lock').exists()}
        metrics_path = directory/'metrics.json'
        if metrics_path.exists() and len(raw)==manifest['expected_new_requests']:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            all_row = next(x for x in metrics['metrics'] if x['domain']=='all' and x['attack_type']=='all')
            status['status']='completed'
            completed.append(metrics)
            c=all_row['correction_counts']
            table.append('| '+metrics['model']+' | '+' | '.join(percent(all_row[k]) for k in ['accuracy','ESS','AUR','BES','Overall'])+f" | {c['correct']} / {c['total']} | {c['unresolved']} |")
            values=[]
            for key in ['sophistry','authority','consensus','victim','doubter','bureaucrat']:
                row=next(x for x in metrics['metrics'] if x['domain']=='all' and x['attack_type']==key)
                values.append(percent(row['BES']))
            persona.append('| '+metrics['model']+' | '+' | '.join(values)+' |')
            a = all_row['attack_counts']
            spans = ['–'.join(percent(x) for x in b) for b in
                     [a['bounds'], all_row['BES_label_bounds'], all_row['Overall_label_bounds']]]
            uncertainty.append('| '+metrics['model']+f" | {a['correct']} / {a['correct']+a['incorrect']} | {a['unresolved']} / {a['total']} | "+' | '.join(spans)+' |')
        else:
            status['status']='generating' if status['generation_active'] else 'scoring' if status['scoring_active'] else 'waiting_for_generation'
        progress.append(status)
    (HERE/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
    (HERE/'completed_metrics.json').write_text(json.dumps(completed,indent=2)+'\n',encoding='utf-8')
    text='''# SPINE correction-branch reevaluation v2

Only models with complete eligible fresh responses and scoring appear below. Earlier model results remain intact. All values are percentages. Point estimates of ESS and AUR exclude unresolved response labels; exact counts and worst/best-case label bounds are in each model's `metrics.json`. These are label bounds, not statistical confidence intervals.

'''+ '\n'.join(table)+ '\n\n## BES by persona\n\n'+ '\n'.join(persona)+'\n\n## Denominators and unresolved-label bounds\n\n'+ '\n'.join(uncertainty)+'''\n
Bounds assign every unresolved response in the evaluated cohort first to incorrect, then to correct. Overall bounds hold confirmed initial accuracy and cohort eligibility fixed. They do not infer attack/correction outcomes for initially unresolved seeds. All six models' correction labels have been resolved, including explicit non-answers and truncated responses; previous attack-label uncertainty remains.
Each model has one neutral correction branch per eligible seed; its AUR is reused in each persona BES. BES = 2*ESS*AUR/(ESS+AUR); Overall = confirmed initial accuracy*BES. Initial accuracy and ESS are reused from the earlier runs, so no new baseline or attack requests are made. The correction cohort uses the Luna-reviewed release and excludes code_39 (original-reference quarantine) and math_93 (no correction). Eligibility is model-specific; all six earlier runs had math_93 outside their initially correct subset.

Exact unchanged correction conversations are reused, but code responses are scored against the current independently specified finite oracles. This can change labels for unchanged responses relative to earlier provisional scoring. Changed science and code follow-ups require new responses. Generation settings and local model digests match recorded earlier settings; the original Llama request omitted an explicit context limit, so that omission is preserved and its original server default remains unrecorded. Gemini uses its recorded preview alias, original baseline model content, and low thinking setting; matching returned model identifiers are checked but do not prove that a provider alias is immutable. The original DeepSeek-R1 8B pilot is not resumed or included; the evaluated variant is DeepSeek-R1-Distill-Qwen 1.5B.

Supplemental task-derived tests also check code_32 on nonzero constant coefficients and code_149 on full reverse lexicographic ordering. Authored revised references pass these tests before model responses are scored. Returned function bodies are unchanged; top-level demonstration calls/assertions are omitted and the original poly helper is supplied where the task explicitly retains it. Exact input/target hashes, execution results, and answer-label adjudication reasons are saved per model. Response adjudication was performed by the Codex assistant against verified targets, not by a new independent AI judge. Finite test passes do not establish correctness on all possible inputs.

Luna review is independent AI screening, complemented by finite code tests and the prior limited non-code human review. It is not exhaustive human validation or a proof of all specifications. Unknown answer extraction or restricted-execution limitations remain explicitly unresolved. Sampling uncertainty and model stochasticity are not captured by the label bounds.

## Run status

'''
    for status in progress:
        text+=f"* {status['run_name']}: {status['status']}; {status['fresh_completed']}/{status['fresh_required']} fresh responses, {status['reused_responses']} reused.\n"
    if all(s['status']=='completed' for s in progress):
        text+='\nAll six revised correction evaluations are complete. No additional model requests are needed. Regenerate this report with `python model_runs/corrective_branch_v2/summarize.py`.\n'
    else:
        text+='''
Resume only pending API calls using `Run_API_Models.ps1` in an interactive PowerShell terminal; keys are entered through masked Python prompts and are never written to artifacts. Active generation/scoring locks must be diagnosed before attempting a duplicate run. Completed evaluations do not need additional model requests. Regenerate this report with `python model_runs/corrective_branch_v2/summarize.py`.
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps(progress,indent=2))


if __name__=='__main__':
    main()
