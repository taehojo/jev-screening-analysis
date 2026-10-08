#!/usr/bin/env python
# AN-0001-08 figures: recall versus proportion of records read (macro-average per collection, and per review) for the ASReview
# default order (mean of seeds 0-9), the hybrid order and the label-free Jev ranking (mean of tie-break seeds 0-9), with the
# tau = 0.07 threshold point and the statistical stop on the Jev order marked. Inputs: recall_curves_macro.csv,
# recall_curves_per_review.csv, per_review_stopping.csv (this folder). Outputs: fig_recall_curves.{png,pdf},
# fig_recall_curves_per_review_{heldout,clef}.png.
import os, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'asr': '#2a78d6', 'hybrid': '#eb6834', 'jev': '#1baf7a'}   # categorical slots 1-3 of the validated default palette (dataviz skill)
LS = {'asr': '-', 'hybrid': '--', 'jev': ':'}
LAB = {'asr': 'ASReview default (mean of 10 seeds)', 'hybrid': 'Hybrid (ASReview + Jev, lambda = 3)', 'jev': 'Jev ranking alone (label-free)'}
TITLE = {'heldout': 'Held-out SYNERGY reviews (n = 23)', 'clef': 'CLEF 2019 Cochrane reviews (n = 28)'}
INK = '#1f2328'; MUTED = '#6b7280'; GRIDC = '#e5e7eb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.titlecolor': INK})

M = pd.read_csv(os.path.join(HERE, 'recall_curves_macro.csv')); R = pd.read_csv(os.path.join(HERE, 'recall_curves_per_review.csv')); S = pd.read_csv(os.path.join(HERE, 'per_review_stopping.csv'))
xcols = [c for c in R.columns if c.startswith('x')]; xs = np.array([float(c[1:]) for c in xcols])

def style(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, color=GRIDC, linewidth=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150)
for ax, coll in zip(axes, ['heldout', 'clef']):
    m = M[M.collection == coll]; s = S[S.collection == coll]
    for k in ['asr', 'hybrid', 'jev']:
        ax.plot(m.proportion_read, m[k], color=COL[k], linestyle=LS[k], linewidth=2, label=LAB[k])
    tx, ty = s.thr_work.mean(), s.thr_rec.mean(); sx, sy = s.jev_stat_work.mean(), s.jev_stat_rec.mean()
    ax.plot([tx], [ty], marker='o', markersize=8, color=COL['jev'], markeredgecolor='white', markeredgewidth=1.5, linestyle='none', label='tau = 0.07 threshold on the Jev ranking (mean over reviews)')
    ax.plot([sx], [sy], marker='s', markersize=7, color=COL['jev'], markeredgecolor='white', markeredgewidth=1.5, linestyle='none', label='Statistical stop on the Jev ranking (mean over reviews)')
    ax.annotate(f'tau = 0.07\n({tx:.2f}, {ty:.3f})', (tx, ty), xytext=(8, -28), textcoords='offset points', fontsize=7.5, color=INK)
    ax.annotate(f'statistical stop\n({sx:.2f}, {sy:.3f})', (sx, sy), xytext=(8, -30), textcoords='offset points', fontsize=7.5, color=INK)
    ax.text(0.99, 0.955, '95% recall', ha='right', va='bottom', fontsize=7.5, color=MUTED)
    style(ax); ax.set_title(TITLE[coll], fontsize=10, loc='left'); ax.set_xlabel('Proportion of records read'); ax.set_ylabel('Recall of included studies (mean over reviews)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_recall_curves.png'), bbox_inches='tight'); fig.savefig(os.path.join(HERE, 'fig_recall_curves.pdf'), bbox_inches='tight'); plt.close(fig)

for coll in ['heldout', 'clef']:
    s = S[S.collection == coll].sort_values('review'); n = len(s); nc = 6; nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.1 * nr), dpi=130, sharex=True, sharey=True); axes = axes.ravel()
    for ax, (_, row) in zip(axes, s.iterrows()):
        for k in ['asr', 'hybrid', 'jev']:
            r = R[(R.collection == coll) & (R.review == row.review) & (R.method == k)]
            ax.plot(xs, r[xcols].values.ravel(), color=COL[k], linestyle=LS[k], linewidth=1.6, label=LAB[k])
        ax.plot([row.thr_work], [row.thr_rec], marker='o', markersize=6, color=COL['jev'], markeredgecolor='white', markeredgewidth=1, linestyle='none', label='tau = 0.07 threshold')
        ax.plot([row.jev_stat_work], [row.jev_stat_rec], marker='s', markersize=5, color=COL['jev'], markeredgecolor='white', markeredgewidth=1, linestyle='none', label='statistical stop, Jev order')
        style(ax); ax.set_title(f'{row.review}\nN = {int(row.N)}, included {int(row.n1)}', fontsize=7, loc='left')
        ax.tick_params(labelsize=7)
    for ax in axes[n:]: ax.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel('Proportion of records read', fontsize=9, y=0.055); fig.supylabel('Recall of included studies', fontsize=9)
    fig.tight_layout(rect=(0.01, 0.07, 1, 1))
    fig.savefig(os.path.join(HERE, f'fig_recall_curves_per_review_{coll}.png'), bbox_inches='tight'); plt.close(fig)
print('figures written')
