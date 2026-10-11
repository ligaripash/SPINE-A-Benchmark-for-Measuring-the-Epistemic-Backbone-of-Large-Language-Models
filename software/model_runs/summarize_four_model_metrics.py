"""Write the SPINE comparison table for models with completed scores."""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (
    ('Llama 3 (8B)', HERE / 'llama3_8b_revised' / 'composite_metrics_by_attack_type.json'),
    ('Qwen 2.5 (7B)', HERE / 'qwen2_5_7b' / 'composite_metrics_by_attack_type.json'),
    ('Mistral Nemo (12B)', HERE / 'mistral_nemo' / 'composite_metrics_by_attack_type.json'),
    ('DeepSeek-R1-Distill-Qwen (1.5B)', HERE / 'deepseek_r1_1_5b' / 'composite_metrics_by_attack_type.json'),
    ('GPT-4o-mini', HERE / 'gpt_4o_mini' / 'composite_metrics_by_attack_type.json'),
    ('Gemini 3 Flash Preview', HERE / 'gemini_3_flash' / 'composite_metrics_by_attack_type.json'),
)
PERSONAS = ('all', 'sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')


def pct(value):
    return '—' if value is None else f'{value * 100:.1f}%'


def get_value(row, new, old=None):
    return row.get(new, row.get(old))


def main():
    missing = [name for name, path in MODELS if not path.exists()]
    results = [(name, json.loads(path.read_text(encoding='utf-8')))
               for name, path in MODELS if path.exists()]
    if not results:
        raise RuntimeError('No completed model metrics')
    hashes = {data['dataset_sha256'] for _, data in results}
    if len(hashes) != 1:
        raise RuntimeError('Runs used different dataset files')
    lines = ['# SPINE comparison: exploratory answer-level metrics', '',
             f'Completed models: {len(results)}. ' +
             ('Pending: ' + ', '.join(missing) + '.' if missing else 'All listed models complete.'),
             '',
             'All models were tested on the same 757 retained seeds and six false-claim attack personas. '
             'ESS and AUR condition on each model’s initially correct anchors; their denominators therefore '
             'differ by model. BES is the harmonic mean of ESS and AUR, and Overall is initial accuracy × BES.',
             '',
             'These are **exploratory, not official benchmark scores**. Most corrective interventions are '
             'candidate drafts without complete independent validation. Code answers were checked with '
             'finite tests. Unknown answer labels are omitted from the point estimates and listed below.',
             '']
    for persona in PERSONAS:
        title = 'All attacks' if persona == 'all' else persona.title()
        lines += [f'## {title}', '',
                  '| Model | Initial accuracy | ESS | Attack unresolved | AUR | Correction unresolved | BES | Overall |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name, data in results:
            row = next(x for x in data['metrics'] if x['attack_type'] == persona)
            initial = get_value(row, 'initial_accuracy', 'initial_accuracy_resolved_only')
            ess = row['ess_resolved_only']
            aur = get_value(row, 'aur_resolved_only_provisional', 'aur_exploratory')
            bes = get_value(row, 'bes_resolved_only_provisional', 'bes_exploratory')
            overall = get_value(row, 'overall_resolved_only_provisional', 'overall_exploratory')
            attack_unknown = (row['attack_counts']['unresolved'] if 'attack_counts' in row
                              else row['attack_unresolved'])
            correction_unknown = (row['correction_counts']['unresolved'] if 'correction_counts' in row
                                  else 0)
            lines.append(f'| {name} | {pct(initial)} | {pct(ess)} | {attack_unknown} | '
                         f'{pct(aur)} | {correction_unknown} | {pct(bes)} | {pct(overall)} |')
        lines.append('')
    lines += ['For exact numerators, denominators, unresolved-label bounds, model identities, and '
              'generation settings, see each model’s `manifest.json` and '
              '`composite_metrics_by_attack_type.json`.', '']
    path = HERE / 'COMPARISON_METRICS.md'
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(path)
    print(path)


if __name__ == '__main__':
    main()
