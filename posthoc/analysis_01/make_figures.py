# AN-0001-01 figures: calibration plots (pooled deciles with Wilson CIs; per-review mean predicted versus observed rate).
import json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(f'{OUT}/results.json'))
C1, C2, C3, CREF = '#1f6fb2', '#c95f02', '#6b4fbb', '#6b6b6b'   # palette validated with the dataviz validator (light surface)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
def panel(ax, key, title, color, per_review=True, label=None):
    v = R['datasets'][key]; D = pd.DataFrame(v['deciles'])
    ax.plot([0, 1], [0, 1], color=CREF, lw=0.7, ls='--')
    if per_review and os.path.exists(f'{OUT}/per_review_{key}.csv'):
        P = pd.read_csv(f'{OUT}/per_review_{key}.csv'); ax.scatter(P.mean_predicted, P.observed_rate, s=np.clip(P.n / 40, 6, 60), facecolors='none', edgecolors=color, linewidths=0.6, alpha=0.7, label='per review (mean predicted vs observed rate)')
    ax.errorbar(D.mean_p, D.obs, yerr=[np.maximum(0, D.obs - D.obs_lo), np.maximum(0, D.obs_hi - D.obs)], color=color, lw=1.4, marker='o', ms=3.5, capsize=2, label=label or 'pooled deciles of p (Wilson 95% CI)')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal'); ax.grid(True, color='#e6e6e6', lw=0.5)
    s = f"slope {v['recal_slope']:.2f}, intercept {v['recal_intercept']:.2f}\nO:E {v['O_E_ratio']:.2f}, ECE {v['ece_10_equal_width']:.3f}\nmean p {v['mean_predicted']:.3f}, observed {v['observed_rate']:.3f}".replace('.', '·')
    ax.text(0.03, 0.97, s, transform=ax.transAxes, va='top', fontsize=6.5); ax.set_title(title, fontsize=8.5)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
panel(axes[0, 0], 'dev_final_batched', 'Development, 73 reviews\nfinal inclusion', C1); panel(axes[0, 1], 'heldout_final_batched', 'Held-out, 23 reviews\nfinal inclusion', C1); panel(axes[0, 2], 'clef_final_batched', 'Cochrane, 28 reviews\nfinal inclusion', C1)
panel(axes[1, 0], 'dev_ta_batched', 'Development, 33 reviews\ntitle and abstract labels', C2); panel(axes[1, 1], 'heldout_ta_batched', 'Held-out, 12 reviews\ntitle and abstract labels', C2); panel(axes[1, 2], 'clef_ta_batched', 'Cochrane, 31 reviews\ntitle and abstract labels', C2)
for ax in axes[1]: ax.set_xlabel('Mean predicted probability')
for ax in axes[:, 0]: ax.set_ylabel('Observed proportion positive')
h, l = axes[0, 2].get_legend_handles_labels(); fig.legend(h, l, fontsize=6.5, frameon=False, loc='lower center', ncol=2); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}/fig_calibration.png', dpi=300); fig.savefig(f'{OUT}/fig_calibration.pdf'); plt.close(fig)
# batched versus single-record scoring on the 2002-record subset, and the pilot review
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
ax = axes[0]; ax.plot([0, 1], [0, 1], color=CREF, lw=0.7, ls='--')
for key, color, lab in [('subset2002_final_single', C1, 'single-record requests'), ('subset2002_final_b10_full', C2, 'ten-record batches (full run)'), ('subset2002_final_b10_subset', C3, 'ten-record batches (within subset)')]:
    D = pd.DataFrame(R['datasets'][key]['deciles']); ax.errorbar(D.mean_p, D.obs, yerr=[np.maximum(0, D.obs - D.obs_lo), np.maximum(0, D.obs_hi - D.obs)], color=color, lw=1.2, marker='o', ms=3, capsize=1.5, label=lab)
ax.set_title('2002-record held-out subset,\nfinal inclusion (prevalence 29·8%)', fontsize=8); ax.legend(fontsize=5.8, frameon=False, loc='upper left', handlelength=1.5); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed proportion positive')
for ax, level, ttl in [(axes[1], 'final', 'Pilot review, final inclusion (77 of 1572)'), (axes[2], 'ta', 'Pilot review, passed title and abstract\nscreening (144 of 1572)')]:
    ax.plot([0, 1], [0, 1], color=CREF, lw=0.7, ls='--')
    for mode, color, lab in [('single', C1, 'single-record requests'), ('batched', C2, 'ten-record batches')]:
        v = R['datasets'][f'pilot_{level}_{mode}']; D = pd.DataFrame(v['deciles']); ax.errorbar(D.mean_p, D.obs, yerr=[np.maximum(0, D.obs - D.obs_lo), np.maximum(0, D.obs_hi - D.obs)], color=color, lw=1.2, marker='o', ms=3, capsize=1.5, label=lab); ax.text(0.03, 0.80 - 0.09 * (mode == 'batched'), f"{lab}: slope {v['recal_slope']:.2f}, O:E {v['O_E_ratio']:.2f}".replace('.', '·'), transform=ax.transAxes, fontsize=5.8, color=color)
    ax.set_title(ttl, fontsize=8); ax.legend(fontsize=5.8, frameon=False, loc='upper left', handlelength=1.5); ax.set_xlabel('Mean predicted probability')
for ax in axes: ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal'); ax.grid(True, color='#e6e6e6', lw=0.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_calibration_batching_pilot.png', dpi=300); fig.savefig(f'{OUT}/fig_calibration_batching_pilot.pdf'); plt.close(fig); print('figures written')
