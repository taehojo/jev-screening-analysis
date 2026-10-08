# 원고 그림 2, 3 (SYNERGY 평가 분할). 모든 값은 결과 파일에서 읽는다.
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.labelcolor': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 300})
mid = FuncFormatter(lambda v, p: (f'{v:.2f}' if abs(v) < 10 else f'{v:.0f}').replace('.', '·'))
OUT = '../manuscript/figures/'
R = json.load(open('test_records.json')); J = {d['id']: d['p'] for d in json.load(open('jev_test.json')) if d.get('ok')}
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
rows = []
for k, v in by.items():
    y = np.array([r['label'] for r in v]); S = json.load(open(f'scores_zs/test_{k}.json'))
    p = np.array([J[r['id']] for r in v]); s = p >= 0.07
    rows.append({'review': k.replace('_', ' '), 'jev': roc_auc_score(y, p), 'bge': roc_auc_score(y, S['bge']), 'thr_work': s.mean(), 'thr_rec': y[s].sum() / y.sum()})
T = pd.DataFrame(rows)
A = pd.DataFrame([d for d in json.load(open('asr_test.json')) if d.get('wss95') is not None])
g = A.groupby(['review', 'method'])[['wss95', 'knee_work', 'knee_rec']].mean().unstack('method'); g.index = g.index.str.replace('_', ' ')
T = T.set_index('review').join(pd.DataFrame({'asr': g[('wss95', 'asr_prior')], 'h3': g[('wss95', 'h3_prior')], 'hyb': g[('wss95', 'asr_jevblend_3')],
                                            'knee_work': g[('knee_work', 'asr_prior')], 'knee_rec': g[('knee_rec', 'asr_prior')]}))
F = json.load(open('final_test.json'))
# ---------------- Figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})
t = T.sort_values('jev'); yy = np.arange(len(t))
a.hlines(yy, t.bge, t.jev, color='#c9c8c3', lw=1.2, zorder=1)
a.scatter(t.bge, yy, s=22, marker='o', color=ORANGE, edgecolor='#fcfcfb', lw=0.6, zorder=2, label='bge-base embedding (best free ranker)')
a.scatter(t.jev, yy, s=26, marker='D', color=BLUE, edgecolor='#fcfcfb', lw=0.6, zorder=3, label='Jev')
a.set_yticks(yy); a.set_yticklabels(t.index, fontsize=6.2); a.set_xlim(0.45, 1.01); a.xaxis.set_major_formatter(mid)
a.set_xlabel('AUC within review'); a.set_title('A  Held-out SYNERGY reviews (n=23)', loc='left', fontsize=8, color=INK, fontweight='bold')
a.legend(loc='upper left', fontsize=6.5, handletextpad=0.3, borderaxespad=0.2); a.grid(axis='x', color='#e6e5e0', lw=0.5); a.set_axisbelow(True)
cost = {'Jev': 0.0201, 'GPT-4o-mini': 0.112, 'DeepSeek-V3.1': 0.160, 'Claude Opus 5.5': 14.65}
auc = {'Jev': F['C2_macro_auc']['jev'], 'GPT-4o-mini': F['C2_macro_auc']['gpt4omini_lp'], 'DeepSeek-V3.1': F['C2_macro_auc']['deepseek'], 'Claude Opus 5.5': F['C2_macro_auc']['claude_opus']}
for m in cost:
    c = BLUE if m == 'Jev' else GRAY; mk = 'D' if m == 'Jev' else 'o'
    b.scatter(cost[m], auc[m], s=40, marker=mk, color=c, edgecolor='#fcfcfb', lw=0.6, zorder=3)
    dx = {'Jev': 1.25, 'GPT-4o-mini': 1.25, 'DeepSeek-V3.1': 1.25, 'Claude Opus 5.5': 0.62}[m]; ha = 'right' if m == 'Claude Opus 5.5' else 'left'
    b.annotate(f"{m}\n" + f"{auc[m]:.3f}".replace('.', '·'), (cost[m], auc[m]), xytext=(cost[m] * dx, auc[m] - 0.002), fontsize=6.5, color=INK, ha=ha, va='center')
b.set_xscale('log'); b.set_xlim(0.01, 40); b.set_ylim(0.9, 0.98); b.yaxis.set_major_formatter(mid)
b.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'${v:g}'.replace('.', '·')))
b.set_xlabel('Cost per 1000 records (US$, log scale)'); b.set_ylabel('Mean AUC across reviews')
b.set_title('B  LLM subset (1999 records)', loc='left', fontsize=8, color=INK, fontweight='bold'); b.grid(color='#e6e5e0', lw=0.5); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT + 'figure2.png'); fig.savefig(OUT + 'figure2.pdf'); plt.close(fig)
# ---------------- Figure 3
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})
t = T.sort_values('hyb'); yy = np.arange(len(t))
lo = t[['asr', 'h3', 'hyb']].min(axis=1); hi = t[['asr', 'h3', 'hyb']].max(axis=1)
a.hlines(yy, lo, hi, color='#c9c8c3', lw=1.2, zorder=1)
a.scatter(t.asr, yy, s=20, marker='o', color=ORANGE, edgecolor='#fcfcfb', lw=0.6, zorder=2, label='ASReview default (TF-IDF + SVM)')
a.scatter(t.h3, yy, s=20, marker='s', color=AQUA, edgecolor='#fcfcfb', lw=0.6, zorder=2, label='ASReview strongest preset (post hoc)')
a.scatter(t.hyb, yy, s=26, marker='D', color=BLUE, edgecolor='#fcfcfb', lw=0.6, zorder=3, label='ASReview + Jev (hybrid)')
a.set_yticks(yy); a.set_yticklabels(t.index, fontsize=6.2); a.set_xlim(0, 1); a.xaxis.set_major_formatter(mid)
a.set_xlabel('Work saved over sampling at 95% recall'); a.set_title('A  Active learning, held-out reviews', loc='left', fontsize=8, color=INK, fontweight='bold')
a.legend(loc='upper left', fontsize=6.5, handletextpad=0.3, borderaxespad=0.2); a.grid(axis='x', color='#e6e5e0', lw=0.5); a.set_axisbelow(True)
b.axhline(0.95, color=INK2, lw=0.8, ls=(0, (3, 2))); b.text(0.30, 0.944, '95% recall', fontsize=6.3, color=INK2, va='top')
b.scatter(T.knee_work, T.knee_rec, s=22, marker='o', color=ORANGE, edgecolor='#fcfcfb', lw=0.6, zorder=2, label=f"Knee method on ASReview ({(T.knee_rec >= 0.95).sum()}/23 reliable)")
b.scatter(T.thr_work, T.thr_rec, s=26, marker='D', color=BLUE, edgecolor='#fcfcfb', lw=0.6, zorder=3, label=f"Jev probability ≥0·07 ({(T.thr_rec >= 0.95).sum()}/23 reliable)")
b.set_xlim(0, 1.02); b.set_ylim(0.6, 1.01); b.xaxis.set_major_formatter(mid); b.yaxis.set_major_formatter(mid)
b.set_xlabel('Proportion of records read'); b.set_ylabel('Recall of included studies')
b.set_title('B  Stopping without labels', loc='left', fontsize=8, color=INK, fontweight='bold'); b.legend(loc='lower left', fontsize=6.5, handletextpad=0.3)
b.grid(color='#e6e5e0', lw=0.5); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT + 'figure3.png'); fig.savefig(OUT + 'figure3.pdf'); plt.close(fig)
T.round(4).to_csv(OUT + 'figure2_3_data.csv'); print(T.round(3).to_string())
