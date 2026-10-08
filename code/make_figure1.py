# 그림 1: 연구 설계 흐름도
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE, INK, INK2, EDGE, FILL, FILLB = '#2a78d6', '#0b0b0b', '#52514e', '#b9b8b2', '#f4f3ef', '#e8f0fb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2})
fig, ax = plt.subplots(figsize=(7.1, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')
def box(x, y, w, h, title, body, fill=FILL, edge=EDGE, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fill, ec=edge, lw=0.8))
    ax.text(x + 1.2, y + h - 1.4, title, fontsize=7.6, fontweight='bold', color=tc, va='top')
    ax.text(x + 1.2, y + h - 4.6, body, fontsize=6.3, color=INK2, va='top', linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8, mutation_scale=8))
box(1, 50, 30, 14, 'SYNERGY 3.0', '114 systematic reviews\n168 969 records; 4399 included\nofficial split: 91 development,\n23 held-out')
box(1, 27, 30, 18, 'Development reviews', '73 reviews with ≤2000 records\n52 957 records; 2048 included\nused to choose every setting')
box(35, 27, 29, 18, 'Settings fixed', 'Comparator: bge-base\nHybrid weight λ=3\nThreshold τ=0·07\nTen records per request', fill=FILLB, edge=BLUE)
box(68, 27, 31, 18, 'Held-out reviews', '23 reviews; 33 001 records\n597 included studies\none prespecified evaluation\nC1–C4 and robustness', fill=FILLB, edge=BLUE)
box(35, 50, 64, 14, 'Comparators', 'Free rankers: TF-IDF, BM25, MiniLM, bge-base\nLLMs (2002-record subset): GPT-4o-mini, DeepSeek-V3.1, Claude Opus 5.5\nActive learning: ASReview default model (post hoc: strongest preset,\nLLM-guided weak supervision); stopping: knee method')
box(1, 2, 47, 20, 'External validation: CLEF 2019', '31 Cochrane reviews (eight diagnostic accuracy,\n20 intervention, one prognosis, two qualitative)\n82 418 records; 882 included\ncriteria from Objectives and Selection criteria\nsettings applied unchanged')
box(52, 2, 47, 20, "External validation: Alzheimer's review", 'unpublished review of germline variants and\ndomain-specific cognition\n1572 PubMed records; 111 included studies\nconclusion-level impact of the stopping threshold')
arrow(16, 50, 16, 45.6); arrow(31.6, 36, 34.4, 36); arrow(64.6, 36, 67.4, 36); arrow(67.5, 49.4, 80, 45.6)
arrow(50, 26.4, 26, 22.6); arrow(50, 26.4, 74, 22.6)
fig.savefig('../manuscript/figures/figure1.png', dpi=300, bbox_inches='tight'); fig.savefig('../manuscript/figures/figure1.pdf', bbox_inches='tight')
