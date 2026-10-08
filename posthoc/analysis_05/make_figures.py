# AN-0001-05 figures: reliability, workload and recall of the label-free threshold rule as a function of tau.
import json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(f'{OUT}/tau_curves.csv'); R = json.load(open(f'{OUT}/results.json'))
C_FINAL, C_TA, C_REF = '#1f6fb2', '#c95f02', '#6b6b6b'   # palette validated with the dataviz validator (light surface)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
cols = [('dev', 'Development reviews'), ('heldout', 'Held-out reviews'), ('clef', 'Cochrane (CLEF 2019) reviews')]
fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), sharex=True)
for j, (key, title) in enumerate(cols):
    for level, colr, lab in [('final', C_FINAL, 'final inclusion'), ('ta', C_TA, 'title and abstract labels')]:
        d = T[T.dataset == f'{key}_{level}'].sort_values('tau'); n = int(d.n_reviews.iloc[0]); lab2 = f'{lab} ({n} reviews)'
        ax = axes[0, j]; ax.fill_between(d.tau, d.wilson_lo, d.wilson_hi, color=colr, alpha=0.12, lw=0); ax.plot(d.tau, d.reliability, color=colr, lw=1.6, marker='o', ms=3, label=lab2)
        ax = axes[1, j]; ax.plot(d.tau, d.macro_work, color=colr, lw=1.6, marker='o', ms=3, label=f'{lab}, macro'); ax.plot(d.tau, d.pooled_work, color=colr, lw=1.2, ls='--', label=f'{lab}, pooled')
        ax = axes[2, j]; ax.plot(d.tau, d.mean_recall, color=colr, lw=1.6, marker='o', ms=3, label=f'{lab}, mean'); ax.plot(d.tau, d.min_recall, color=colr, lw=1.2, ls=':', label=f'{lab}, minimum')
    for i in range(3):
        ax = axes[i, j]; ax.axvline(0.07, color=C_REF, lw=0.8, ls='-', alpha=0.7); ax.set_xscale('log'); ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(['0·01', '0·02', '0·05', '0·1', '0·2', '0·5']); ax.grid(True, color='#e6e6e6', lw=0.5); ax.set_ylim(-0.02, 1.02)
    axes[0, j].axhline(0.95, color=C_REF, lw=0.6, ls='--'); axes[0, j].axhline(0.90, color=C_REF, lw=0.6, ls=':'); axes[0, j].set_title(title, fontsize=8.5)
    axes[2, j].set_xlabel('Threshold τ (log scale)')
axes[0, 0].set_ylabel('Reliability (proportion of reviews\nwith recall ≥ 0·95), Wilson 95% CI'); axes[1, 0].set_ylabel('Proportion of records read'); axes[2, 0].set_ylabel('Recall of the rule')
axes[0, 0].text(0.072, 0.03, 'τ=0·07', color=C_REF, fontsize=7); axes[0, 0].text(0.0105, 0.955, '0·95', color=C_REF, fontsize=6.5); axes[0, 0].text(0.0105, 0.905, '0·90', color=C_REF, fontsize=6.5)
axes[0, 2].legend(fontsize=6, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.22)); axes[1, 2].legend(fontsize=6, frameon=False, loc='upper right'); axes[2, 2].legend(fontsize=6, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_tau_curves.png', dpi=300); fig.savefig(f'{OUT}/fig_tau_curves.pdf'); plt.close(fig)
# bootstrap re-selection distributions
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, (g, title) in zip(axes, [('original_grid', 'Original grid (as searched)'), ('fine_grid', 'Fine grid (0·01 steps to 0·10)')]):
    dist = R['bootstrap_reselection'][g]['selected_tau_distribution']; xs = [str(d['tau']).replace('.', '·') for d in dist]; ys = [d['proportion'] for d in dist]
    bars = ax.bar(xs, ys, color=C_FINAL, width=0.6); ax.set_title(title, fontsize=8.5); ax.set_xlabel('Re-selected τ'); ax.set_ylabel('Proportion of 5000 resamples'); ax.set_ylim(0, 1)
    for b, y in zip(bars, ys): ax.text(b.get_x() + b.get_width() / 2, y + 0.02, f'{y:.3f}'.replace('.', '·'), ha='center', fontsize=7)
    ax.grid(True, axis='y', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_tau_reselection.png', dpi=300); fig.savefig(f'{OUT}/fig_tau_reselection.pdf'); plt.close(fig)
print('figures written')
