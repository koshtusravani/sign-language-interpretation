import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('results', exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
})

epochs = list(range(1, 41))
losses = [
    54.4828, 43.2580, 40.2060, 38.6205, 31.0512, 34.4976, 30.1842, 30.0468,
    27.6548, 26.7309, 21.4408, 20.2526, 17.8389, 16.8610, 17.1059, 18.4546,
    17.1052, 19.4572, 15.1245, 14.8798, 14.9134, 13.9955, 17.8524, 15.5564,
    15.9002, 13.9864, 15.0574, 13.3457, 13.5208, 14.1566, 12.1041, 12.4847,
    13.1985, 12.0776, 12.8881, 11.9285, 14.2349, 11.8855, 14.8216, 12.6531,
]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, losses, color='#2E75B6', linewidth=1.8, zorder=3)
ax.fill_between(epochs, losses, alpha=0.08, color='#2E75B6')
ax.axhline(y=11.8855, color='#C00000', linestyle='--', linewidth=1.2,
           label='Best loss: 11.89 (epoch 38)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Figure 1. ResNet18 Training Loss over 40 Epochs')
ax.legend(fontsize=10)
ax.set_xlim(1, 40)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/fig1_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig1_training_loss.png")

labels   = ['Greedy\n(no context)', 'HMM\nuni-trans', 'HMM\nest+avg', 'HMM\nest+conf']
slot_acc = [0.5622, 0.5622, 0.6067, 0.6567]
seq_acc  = [0.1533, 0.1533, 0.4533, 0.4967]

x     = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars1 = ax.bar(x - width / 2, slot_acc, width,
               label='Slot-level accuracy', color='#2E75B6', zorder=3)
bars2 = ax.bar(x + width / 2, seq_acc,  width,
               label='Sequence-level accuracy', color='#ED7D31', zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy')
ax.set_title('Figure 2. Slot-Level and Sequence-Level Accuracy by Decoding Method')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 0.80)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/fig2_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig2_accuracy_comparison.png")

classes   = ['basketball', 'birthday', 'but', 'city', 'man',
             'many', 'orange', 'play', 'shirt', 'who']
precision = [0.50, 0.00, 1.00, 1.00, 0.50, 0.25, 0.33, 0.67, 1.00, 0.67]
recall    = [0.33, 0.00, 0.50, 1.00, 0.67, 0.50, 0.50, 0.67, 0.67, 0.67]
f1        = [0.40, 0.00, 0.67, 1.00, 0.57, 0.33, 0.40, 0.67, 0.80, 0.67]

x     = np.arange(len(classes))
width = 0.28

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - width, precision, width, label='Precision', color='#2E75B6', zorder=3)
ax.bar(x,         recall,    width, label='Recall',    color='#ED7D31', zorder=3)
ax.bar(x + width, f1,        width, label='F1-Score',  color='#70AD47', zorder=3)
ax.set_ylabel('Score')
ax.set_title('Figure 3. Per-Class Precision, Recall, and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/fig3_per_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig3_per_class.png")

categories = [
    'HMM corrected\nCNN error',
    'Both correct\n(CNN + HMM)',
    'HMM degraded\nCNN',
    'Both wrong\n(CNN + HMM)',
]
counts     = [11, 13, 14, 32]
mean_ents  = [1.9990, 1.8522, 1.9838, 2.0194]
colors     = ['#70AD47', '#2E75B6', '#ED7D31', '#C00000']
max_ent    = 2.3026

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

bars = ax1.bar(categories, counts, color=colors, zorder=3, width=0.55)
for bar, c in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             str(c), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_ylabel('Number of word slots')
ax1.set_title('(a) Outcome counts  (total = 70)')
ax1.set_ylim(0, 42)
ax1.tick_params(axis='x', labelsize=8.5)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

bars2 = ax2.bar(categories, mean_ents, color=colors, zorder=3, width=0.55)
ax2.axhline(y=max_ent, color='gray', linestyle='--', linewidth=1.0,
            label=f'Max entropy ({max_ent:.4f})')
for bar, e in zip(bars2, mean_ents):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{e:.3f}', ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('Mean Shannon entropy (nats)')
ax2.set_title('(b) Mean entropy per outcome')
ax2.set_ylim(1.6, 2.45)
ax2.legend(fontsize=9)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle(
    'Figure 4. Outcome Counts and Mean Prediction Entropy by Decoding Result',
    fontsize=11, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.savefig('results/fig4_entropy_outcomes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig4_entropy_outcomes.png")

print("\nAll four figures saved to results/")