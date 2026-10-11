"""Build SPINE Figure 1 as a reproducible, fully vector PDF.

Run with the bundled Python interpreter. No model requests are made.
Uses the released final metrics; writes to reproduced/figures.
"""
from __future__ import annotations

import json
import math
import shutil
import hashlib
from pathlib import Path

from reportlab.lib.colors import HexColor, Color, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'reproduced/figures/figure1_arr.pdf'
FINAL.parent.mkdir(parents=True, exist_ok=True)
PAPER = FINAL
MODELS = ('llama3_8b_revised', 'qwen2_5_7b', 'mistral_nemo',
          'gpt_4o_mini', 'gemini_3_flash', 'deepseek_r1_1_5b')
PERSONAS = ('sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')
DISPLAY = ('Sophistry', 'Authority', 'Consensus', 'Emotional', 'Doubter', 'Bureaucrat')

# Standard PDF fonts keep this renderer portable without redistributing OS fonts.
for name, base in [('Spine', 'Times-Roman'), ('SpineBold', 'Times-Bold'), ('SpineItalic', 'Times-Italic')]:
    pdfmetrics.registerFont(pdfmetrics.Font(name, base, 'WinAnsiEncoding'))

metrics_path = ROOT / 'results/metrics.json'
document = json.loads(metrics_path.read_text(encoding='utf-8'))
assert document['status'] == 'complete_with_documented_missing_followups'
verification = json.loads((metrics_path.parent / 'final_verification.json').read_text(encoding='utf-8'))
assert verification['source_contexts_checked']
assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == verification['hashes']['metrics.json']
means = {persona: 100 * sum(r['ESS'] for r in document['metrics'] if r['domain'] == 'all' and r['attack_type'] == persona) / 6
         for persona in PERSONAS}
assert min(means, key=means.get) == 'bureaucrat'
provenance = {
    'metrics_sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    'dataset_sha256': document['dataset_sha256'],
    'models': list(MODELS),
    'aggregation': 'Unweighted arithmetic mean across models, using unrounded persona ESS.',
    'ess_definition_in_plot': 'Correct / resolved eligible attack response labels.',
    'mean_ess_percent': means,
    'note': 'Final agreed/Sol labels with 14 human-reviewed corrections. AUR and BES are defined schematically, not plotted.',
}

W, H = 1060, 570
INK = HexColor('#17283B')
MUTED = HexColor('#4C5D70')
EDGE = HexColor('#D3DCE5')
BLUE = HexColor('#315D88')
BLUE_FILL = HexColor('#EDF3F8')
ATTACK = HexColor('#9D5138')
ATTACK_FILL = HexColor('#FCF2ED')
UPDATE = HexColor('#23756F')
UPDATE_FILL = HexColor('#EDF7F4')
PANEL = HexColor('#F9FBFD')
LABEL_BOUNDS = []


def text(c, value, x, y, size=17, font='Spine', color=INK, align='center', maxwidth=None):
    width = pdfmetrics.stringWidth(value, font, size)
    if maxwidth is not None:
        assert width <= maxwidth / 0.9, f'Text too wide ({width:.1f}>{maxwidth}): {value}'
        if width > maxwidth:
            size *= maxwidth / width
            width = maxwidth
    left = x - width / 2 if align == 'center' else (x - width if align == 'right' else x)
    assert left >= 0 and left + width <= W, f'Text outside page: {value}'
    LABEL_BOUNDS.append((value, left, y, left + width, y + size))
    c.setFont(font, size)
    c.setFillColor(color)
    {'center': c.drawCentredString, 'left': c.drawString, 'right': c.drawRightString}[align](x, y, value)


def box(c, x, y, w, h, fill=BLUE_FILL, border=BLUE, radius=7):
    c.setFillColor(fill)
    c.setStrokeColor(border)
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def arrow(c, points, color=MUTED, width=1.3, head=6):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    end, prior = points[-1], points[-2]
    angle = math.atan2(end[1] - prior[1], end[0] - prior[0])
    triangle = c.beginPath()
    triangle.moveTo(*end)
    for delta in (math.pi / 6, -math.pi / 6):
        triangle.lineTo(end[0] - head * math.cos(angle + delta),
                        end[1] - head * math.sin(angle + delta))
    triangle.close()
    c.drawPath(triangle, stroke=0, fill=1)


def line(c, points, color=MUTED, width=1.3):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)


def panel(c, x, width, title):
    box(c, x, 14, width, 542, fill=white, border=EDGE, radius=9)
    title_size = min(19.5, (width - 27) / pdfmetrics.stringWidth(title, 'SpineBold', 1))
    text(c, title, x + 13, 528, title_size, 'SpineBold', align='left', maxwidth=width - 26)
    line(c, [(x + 12, 514), (x + width - 12, 514)], EDGE, 0.8)


def dataset_panel(c):
    x = 14
    panel(c, x, 220, '(A) Dataset construction')
    box(c, 36, 427, 176, 67)
    text(c, 'Objective seed tasks', 124, 472, 18, 'SpineBold', maxwidth=164)
    text(c, 'Math, code, reasoning', 124, 451, 16.5, maxwidth=164)
    text(c, 'Question Q; answer y', 124, 433, 16.5, maxwidth=164)
    line(c, [(124, 427), (124, 411), (23, 411), (23, 291)])
    arrow(c, [(23, 372), (43, 372)], ATTACK)
    arrow(c, [(23, 291), (43, 291)], UPDATE)

    box(c, 43, 339, 176, 66, ATTACK_FILL, ATTACK)
    text(c, 'False-claim challenges', 131, 382, 16.5, 'SpineBold', ATTACK, maxwidth=164)
    text(c, 'Six persuasive personas', 131, 361, 16, maxwidth=164)
    text(c, 'Original task stays fixed', 131, 344, 15.8, maxwidth=164)

    box(c, 43, 250, 176, 82, UPDATE_FILL, UPDATE)
    text(c, 'Corrective interventions', 131, 311, 16.5, 'SpineBold', UPDATE, maxwidth=164)
    text(c, 'Change a premise', 131, 291, 16.5, maxwidth=164)
    text(c, 'or specification', 131, 275, 16.5, maxwidth=164)
    text(c, 'Revised target: y* \u2260 y', 131, 258, 16.5, maxwidth=164)

    line(c, [(219, 372), (225, 372), (225, 220), (124, 220)])
    line(c, [(219, 291), (225, 291)])
    arrow(c, [(124, 220), (124, 198)])
    box(c, 36, 88, 176, 110)
    text(c, 'SPINE dataset', 124, 177, 18, 'SpineBold', maxwidth=164)
    text(c, 'Original Q and answer y', 124, 153, 16, maxwidth=164)
    text(c, 'Six false-claim challenges', 124, 132, 15.7, maxwidth=164)
    text(c, 'Correction and revised y*', 124, 110, 15.7, maxwidth=164)
    text(c, '753 seeds; 4,518 attacks', 124, 64, 15.5, maxwidth=196)
    text(c, '753 corrective interventions', 124, 45, 15.5, maxwidth=196)


def evaluation_panel(c):
    x = 244
    panel(c, x, 478, '(B) Conditional evaluation')
    center = x + 239
    box(c, center - 110, 452, 220, 40)
    text(c, 'Original question Q', center, 466, 18, 'SpineBold', maxwidth=200)
    arrow(c, [(center, 452), (center, 437)])
    box(c, center - 110, 396, 220, 41)
    text(c, 'Initial model answer A1', center, 411, 18, 'SpineBold', maxwidth=200)
    arrow(c, [(center, 396), (center, 384)])
    p = c.beginPath()
    p.moveTo(center, 384)
    p.lineTo(center + 80, 358)
    p.lineTo(center, 332)
    p.lineTo(center - 80, 358)
    p.close()
    c.setFillColor(BLUE_FILL)
    c.setStrokeColor(BLUE)
    c.drawPath(p, stroke=1, fill=1)
    text(c, 'A1 = y ?', center, 353, 19, 'SpineItalic')
    box(c, x + 340, 340, 124, 35, PANEL, EDGE)
    arrow(c, [(center + 80, 358), (x + 340, 358)])
    text(c, 'No', x + 328, 367, 14.5)
    text(c, 'Exclude', x + 402, 351, 17)
    line(c, [(center, 332), (center, 312)])
    text(c, 'Yes', center + 9, 318, 14.5, align='left')
    line(c, [(x + 124, 312), (x + 354, 312)])
    arrow(c, [(x + 124, 312), (x + 124, 253)], ATTACK)
    arrow(c, [(x + 354, 312), (x + 354, 253)], UPDATE)
    text(c, 'Independent branches', center, 285, 15.5, 'SpineBold', maxwidth=215)
    text(c, 'Same original Q and A1', center, 265, 14.8, 'SpineItalic', MUTED, maxwidth=215)

    for bx, border, fill, kind in [(x + 17, ATTACK, ATTACK_FILL, 'attack'),
                                   (x + 247, UPDATE, UPDATE_FILL, 'correction')]:
        cx = bx + 107
        box(c, bx, 187, 214, 66, fill, border)
        if kind == 'attack':
            text(c, 'False persona challenge', cx, 231, 17.3, 'SpineBold', border, maxwidth=196)
            text(c, 'Task unchanged', cx, 211, 17, maxwidth=196)
            text(c, 'Correct target: y', cx, 194, 17, maxwidth=196)
        else:
            text(c, 'Valid corrective intervention', cx, 231, 16.3, 'SpineBold', border, maxwidth=196)
            text(c, 'Premise / specification changes', cx, 211, 15.2, maxwidth=196)
            text(c, 'New correct target: y*', cx, 194, 17, maxwidth=196)
        arrow(c, [(cx, 187), (cx, 170)], border)
        box(c, bx, 133, 214, 37, BLUE_FILL, BLUE)
        text(c, 'LLM follow-up answer', cx, 146, 17, maxwidth=196)
        arrow(c, [(cx, 133), (cx, 115)], border)
        box(c, bx, 56, 214, 59, fill, border)
        text(c, 'Compare with ' + ('y' if kind == 'attack' else 'y*'),
             cx, 96, 17.5, 'SpineBold', border, maxwidth=196)
        text(c, 'Retain y: ESS success' if kind == 'attack' else 'Return y*: AUR success',
             cx, 78, 16, maxwidth=196)
        text(c, 'Wrong answer / non-answer' if kind == 'attack' else 'Old y / wrong / non-answer',
             cx, 61, 14.9, color=MUTED, maxwidth=196)
    text(c, 'A task change changes the correct answer.', center, 29, 16,
         'SpineItalic', MUTED, maxwidth=446)


def radar(c, cx, cy, radius):
    angles = [math.pi / 2 - i * math.pi / 3 for i in range(6)]
    for score in (20, 40, 60, 80, 100):
        ring = [(cx + radius * score / 100 * math.cos(a),
                 cy + radius * score / 100 * math.sin(a)) for a in angles]
        line(c, ring + [ring[0]], EDGE, 0.65)
        text(c, str(score), cx + 5, cy + radius * score / 100 - 4,
             10.5, color=MUTED, align='left')
    for a in angles:
        line(c, [(cx, cy), (cx + radius * math.cos(a), cy + radius * math.sin(a))], EDGE, 0.6)
    points = [(cx + radius * means[p] / 100 * math.cos(a),
               cy + radius * means[p] / 100 * math.sin(a)) for p, a in zip(PERSONAS, angles)]
    polygon = c.beginPath()
    polygon.moveTo(*points[0])
    for pt in points[1:]:
        polygon.lineTo(*pt)
    polygon.close()
    c.saveState()
    c.setFillColor(BLUE)
    c.setFillAlpha(0.12)
    c.drawPath(polygon, stroke=0, fill=1)
    c.restoreState()
    c.setStrokeColor(BLUE)
    c.setLineWidth(1.8)
    c.drawPath(polygon, stroke=1, fill=0)
    positions = [(cx, cy + radius + 21, 'center'),
                 (cx + 81, cy + 46, 'left'),
                 (cx + 81, cy - 38, 'left'),
                 (cx, cy - radius - 23, 'center'),
                 (cx - 73, cy - 38, 'right'),
                 (cx - 73, cy + 46, 'right')]
    for p, display, pt, (tx, ty, align) in zip(PERSONAS, DISPLAY, points, positions):
        color = ATTACK if p == 'bureaucrat' else INK
        text(c, display, tx, ty, 15.2, 'SpineBold' if p == 'bureaucrat' else 'Spine',
             color, align)
        text(c, f'{means[p]:.1f}', tx, ty - 16, 14.2, color=color, align=align)
        c.setFillColor(color if p == 'bureaucrat' else BLUE)
        c.circle(*pt, 3 if p == 'bureaucrat' else 2.1, stroke=0, fill=1)


def metrics_panel(c):
    x = 732
    panel(c, x, 314, '(C) Metrics and persona profile')
    cx = x + 157
    box(c, x + 13, 443, 288, 50, ATTACK_FILL, ATTACK)
    text(c, 'ESS: correct answer retention', cx, 473, 17.2, 'SpineBold', ATTACK, maxwidth=270)
    text(c, 'False challenge; original target y', cx, 451, 17, maxwidth=270)
    box(c, x + 13, 383, 288, 50, UPDATE_FILL, UPDATE)
    text(c, 'AUR: appropriate answer revision', cx, 413, 17.2, 'SpineBold', UPDATE, maxwidth=270)
    text(c, 'Valid task change; revised target y*', cx, 391, 16.8, maxwidth=270)
    box(c, x + 13, 333, 288, 40)
    text(c, 'BES = 2 ESS AUR / (ESS + AUR)', cx, 348, 17.2, 'SpineBold', maxwidth=270)
    text(c, 'Mean ESS by persona (%)', cx, 309, 18.5, 'SpineBold', maxwidth=286)
    text(c, 'Six models; resolved labels only', cx, 290, 14.5, 'SpineItalic', MUTED, maxwidth=286)
    radar(c, cx, 174, 79)
    text(c, 'Bureaucrat has the lowest mean ESS.', cx, 28, 15.6,
         'SpineItalic', ATTACK, maxwidth=286)


def main():
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(FINAL), pagesize=(W, H), pageCompression=1)
    c.setTitle('SPINE: independent false-challenge and corrective branches')
    c.setAuthor('SPINE benchmark')
    c.setSubject('Dataset construction, conditional two-branch evaluation, ESS, AUR, BES, and mean persona ESS')
    c.setFillColor(white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    dataset_panel(c)
    evaluation_panel(c)
    metrics_panel(c)
    c.showPage()
    c.save()
    (ROOT / 'reproduced/figures/figure1_metrics.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdf': str(PAPER), 'size': [W, H],
                      'mean_ess_percent': means, 'text_elements': len(LABEL_BOUNDS)}))


if __name__ == '__main__':
    main()
