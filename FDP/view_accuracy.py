"""
Accuracy Visualization — Future Disease Prediction System
Run after training: python view_accuracy.py
Requires: pip install matplotlib
Saves chart to: models/accuracy_chart.png
"""

import json, os, sys

try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend (works without display)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib not found. Install it: pip install matplotlib")
    sys.exit(1)

# ── Load results ─────────────────────────────────────────────
if not os.path.exists('models/accuracy_results.json'):
    print("No accuracy results found. Run train_model_sklearn.py first.")
    sys.exit(1)

with open('models/accuracy_results.json') as f:
    acc = json.load(f)

diseases = list(acc['per_disease'].keys())
scores   = list(acc['per_disease'].values())
overall  = acc.get('overall', 0)

# Friendly labels
labels = {
    'diabetes':             'Diabetes',
    'hypertension':         'Hypertension',
    'heart_disease':        'Heart Disease',
    'hypothyroidism':       'Hypothyroidism',
    'sleep_disorder':       'Sleep Disorder',
    'obesity_risk':         'Obesity Risk',
    'occupational_disease': 'Occupational',
    'respiratory_disease':  'Respiratory',
}
friendly = [labels.get(d, d) for d in diseases]

# Color by accuracy level
bar_colors = []
for s in scores:
    if s >= 95:   bar_colors.append('#2e7d32')   # green — excellent
    elif s >= 90: bar_colors.append('#1565c0')   # blue  — good
    elif s >= 85: bar_colors.append('#f57c00')   # orange — fair
    else:         bar_colors.append('#c62828')   # red   — needs work

# ── Figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#f0f4f8')

# Title
fig.suptitle(
    'Future Disease Prediction System — Model Accuracy Report',
    fontsize=15, fontweight='bold', color='#1a237e', y=0.97
)

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# ── Chart 1: Horizontal bar chart ────────────────────────────
ax1.set_facecolor('#ffffff')
bars = ax1.barh(friendly, scores, color=bar_colors, edgecolor='white',
                linewidth=0.8, height=0.6)
ax1.set_xlim(0, 108)
ax1.set_xlabel('Accuracy (%)', fontsize=10)
ax1.set_title('Per-disease accuracy', fontsize=11, fontweight='bold', color='#1a237e')
ax1.axvline(x=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=overall, color='#c62828', linestyle='-', alpha=0.7, linewidth=1.5,
            label=f'Overall: {overall:.1f}%')
for bar, score in zip(bars, scores):
    ax1.text(score + 0.5, bar.get_y() + bar.get_height()/2,
             f'{score:.1f}%', va='center', fontsize=9, fontweight='bold',
             color='#1a237e')
ax1.legend(fontsize=9)
ax1.tick_params(labelsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Chart 2: Pie chart of risk categories ────────────────────
ax2.set_facecolor('#ffffff')
excellent = sum(1 for s in scores if s >= 95)
good      = sum(1 for s in scores if 90 <= s < 95)
fair      = sum(1 for s in scores if 85 <= s < 90)
needs     = sum(1 for s in scores if s < 85)
pie_vals  = [x for x in [excellent, good, fair, needs] if x > 0]
pie_labels = [l for l, x in zip(
    ['Excellent\n(>=95%)', 'Good\n(90-94%)', 'Fair\n(85-89%)', 'Needs work\n(<85%)'],
    [excellent, good, fair, needs]) if x > 0]
pie_colors = ['#2e7d32', '#1565c0', '#f57c00', '#c62828'][:len(pie_vals)]
wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct='%1.0f%%', startangle=90,
    textprops={'fontsize': 9})
for at in autotexts:
    at.set_fontweight('bold')
ax2.set_title('Accuracy distribution', fontsize=11, fontweight='bold', color='#1a237e')

# ── Chart 3: Training history (if available) ─────────────────
ax3.set_facecolor('#ffffff')
if os.path.exists('models/training_history.csv'):
    import pandas as pd
    history = pd.read_csv('models/training_history.csv')
    epochs  = range(1, len(history) + 1)
    if 'loss' in history.columns:
        ax3.plot(epochs, history['loss'],     color='#c62828', label='Training loss',
                 linewidth=2)
    if 'val_loss' in history.columns:
        ax3.plot(epochs, history['val_loss'], color='#1a237e', label='Validation loss',
                 linewidth=2, linestyle='--')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.set_title('Training vs validation loss', fontsize=11, fontweight='bold', color='#1a237e')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
else:
    ax3.text(0.5, 0.5, 'Training history\nnot available.\n\nRun train_model.py\nfor full TF CNN.',
             ha='center', va='center', transform=ax3.transAxes, fontsize=11,
             color='#607d8b')
    ax3.set_title('Training history', fontsize=11, fontweight='bold', color='#1a237e')
ax3.tick_params(labelsize=9)

# ── Chart 4: Summary stats table ─────────────────────────────
ax4.set_facecolor('#ffffff')
ax4.axis('off')
table_data = [['Disease', 'Accuracy', 'Grade']]
for d, s in zip(friendly, scores):
    grade = 'Excellent' if s >= 95 else 'Good' if s >= 90 else 'Fair' if s >= 85 else 'Review'
    table_data.append([d, f'{s:.1f}%', grade])
table_data.append(['── OVERALL ──', f'{overall:.1f}%', ''])

tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cfd8dc')
    if row == 0:
        cell.set_facecolor('#1a237e')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == len(diseases) + 1:
        cell.set_facecolor('#e8f5e9')
        cell.set_text_props(fontweight='bold', color='#1b5e20')
    elif row % 2 == 0:
        cell.set_facecolor('#f5f5f5')
    else:
        cell.set_facecolor('#ffffff')

ax4.set_title('Accuracy summary', fontsize=11, fontweight='bold', color='#1a237e')

plt.tight_layout(rect=[0, 0, 1, 0.95])

out = 'models/accuracy_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Chart saved: {out}")
print(f"Overall accuracy: {overall}%")
for d, s in zip(friendly, scores):
    grade = 'Excellent' if s >= 95 else 'Good' if s >= 90 else 'Fair' if s >= 85 else 'Review'
    print(f"  {d:22s}: {s:5.1f}%  [{grade}]")
