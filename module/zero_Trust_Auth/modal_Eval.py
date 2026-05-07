# Full evaluation of Isolation Forest + One-Class SVM
# Outputs: precision, recall, F1, ROC curve, confusion matrix

# Saves to:   Eval_results/evaluation_report.png

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed - saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    classification_report
)

# -* Config -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
ROOT_PATH    = Path(__file__).parent.parent.parent
MODELS_DIR   = ROOT_PATH / "models" / "zero_trust_auth"
DATA_FILE    = ROOT_PATH / "module" / "zero_Trust_Auth" / "data_processing" / "user_behavioral_profiles_combined.csv"
RESULTS_DIR  = ROOT_PATH / "module" / "zero_Trust_Auth" / "Eval_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -* Load models and data -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
print("Loading models and data...")

scaler      = joblib.load(MODELS_DIR / "scaler_v2.pkl")
iso_forest  = joblib.load(MODELS_DIR / "isolation_forest_v2.pkl")
oc_svm      = joblib.load(MODELS_DIR / "oneclass_svm_v2.pkl")

df          = pd.read_csv(DATA_FILE)

FEATURE_COLS = [c for c in df.columns if c != 'user']

X_raw        = df[FEATURE_COLS].values
X_scaled     = scaler.transform(X_raw)
print(f"Dataset: {X_scaled.shape[0]} users × {X_scaled.shape[1]} features")

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─
# EVALUATING UNSUPERVISED MODELS
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─
# The BB-MAS dataset has no attacker labels - everyone is a
# legitimate user. So here this create a synthetic test set:
#
#   NORMAL samples  = real user profiles from the dataset (label = 0)
#   ANOMALY samples = randomly perturbed versions of profiles (label = 1)
#                     simulating an attacker who types differently
#
# This is standard practice for evaluating one-class classifiers.
# We create 3 types of anomalies to test different attack scenarios:
#   Type 1 - Large deviation (obvious attacker - different person entirely)
#   Type 2 - Medium deviation (cautious attacker - trying to mimic)
#   Type 3 - Small deviation (very subtle - almost identical but slightly off)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─

print("\nGenerating synthetic anomaly test set...")
np.random.seed(42)

n_users = len(X_scaled)
normal_samples  = X_scaled.copy()
normal_labels   = np.zeros(n_users)   # 0 = normal

# Type 1: Large anomalies - attacker types very differently
# Add large random noise (±3 standard deviations)
anomaly_large = X_scaled + np.random.normal(0, 3.0, X_scaled.shape)
labels_large  = np.ones(n_users)

# Type 2: Medium anomalies - attacker tries to blend in
# Add medium noise (±1.5 std)
anomaly_medium = X_scaled + np.random.normal(0, 1.5, X_scaled.shape)
labels_medium  = np.ones(n_users)

# Type 3: Small anomalies - very subtle behaviour shift
# Add small noise (±0.7 std) - hardest to detect
anomaly_small = X_scaled + np.random.normal(0, 0.7, X_scaled.shape)
labels_small  = np.ones(n_users)

# Full test set: all normals + all 3 anomaly types
X_test = np.vstack([
    normal_samples,
    anomaly_large,
    anomaly_medium,
    anomaly_small
])
y_true = np.concatenate([
    normal_labels,
    labels_large,
    labels_medium,
    labels_small
])

print(f"Test set: {len(X_test)} samples")
print(f"  Normal  : {int(np.sum(y_true==0))}")
print(f"  Anomaly : {int(np.sum(y_true==1))}")

# -* Get predictions from both models -*-*-*-*-*-*-*-*-*-*─
# Models return +1 (normal) or -1 (anomaly)
# We convert to 0 (normal) and 1 (anomaly) to match y_true

print("\nScoring test set with both models...")

# Isolation Forest
if_raw_preds  = iso_forest.predict(X_test)          # +1 or -1
if_preds      = (if_raw_preds == -1).astype(int)    # 1=anomaly, 0=normal
if_scores_raw = iso_forest.score_samples(X_test)    # raw scores for ROC
# Invert: lower score = more anomalous = higher probability
if_scores_roc = -if_scores_raw

# One-Class SVM
svm_raw_preds  = oc_svm.predict(X_test)
svm_preds      = (svm_raw_preds == -1).astype(int)
svm_scores_raw = oc_svm.score_samples(X_test)
svm_scores_roc = -svm_scores_raw

# Combined (layered): IF first, SVM only when IF is suspicious
combined_preds = np.zeros(len(X_test), dtype=int)
for i in range(len(X_test)):
    if if_preds[i] == 1:
        # IF flagged it - confirm with SVM
        combined_preds[i] = svm_preds[i]
    else:
        combined_preds[i] = 0  # IF says normal - trust it

# Combined score for ROC: average of both scores
combined_scores_roc = (if_scores_roc + svm_scores_roc) / 2.0

# -* Compute metrics -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─
def metrics(y_true, y_pred, name):
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{name}")
    print(f"  Precision : {p:.4f}  (of all flagged, how many were real anomalies)")
    print(f"  Recall    : {r:.4f}  (of all anomalies, how many were caught)")
    print(f"  F1 Score  : {f1:.4f}  (balance between precision and recall)")
    return p, r, f1

print("\n" + "="*55)
print("EVALUATION RESULTS")
print("="*55)

if_p,  if_r,  if_f1  = metrics(y_true, if_preds,       "Isolation Forest")
svm_p, svm_r, svm_f1 = metrics(y_true, svm_preds,      "One-Class SVM")
cm_p,  cm_r,  cm_f1  = metrics(y_true, combined_preds, "Combined (Layered IF + SVM)")

# Per-anomaly-type breakdown
print("\n--- Breakdown by anomaly difficulty ---")
n = n_users
for label, start, end in [
    ("Large anomaly  (obvious)",  n,   2*n),
    ("Medium anomaly (moderate)", 2*n, 3*n),
    ("Small anomaly  (subtle)",   3*n, 4*n),
]:
    y_t = y_true[start:end]
    y_p = combined_preds[start:end]
    r   = recall_score(y_t, y_p, zero_division=0)
    print(f"  {label}: detection rate = {r:.1%}")

# ROC curves
if_fpr,  if_tpr,  _ = roc_curve(y_true, if_scores_roc)
svm_fpr, svm_tpr, _ = roc_curve(y_true, svm_scores_roc)
cm_fpr,  cm_tpr,  _ = roc_curve(y_true, combined_scores_roc)
if_auc  = auc(if_fpr,  if_tpr)
svm_auc = auc(svm_fpr, svm_tpr)
cm_auc  = auc(cm_fpr,  cm_tpr)

print(f"\n--- AUC (Area Under ROC Curve) ---")
print(f"  Isolation Forest : {if_auc:.4f}")
print(f"  One-Class SVM    : {svm_auc:.4f}")
print(f"  Combined         : {cm_auc:.4f}")
print(f"  (1.0 = perfect, 0.5 = random guessing)")

# Confusion matrices
cm_if  = confusion_matrix(y_true, if_preds)
cm_svm = confusion_matrix(y_true, svm_preds)
cm_comb= confusion_matrix(y_true, combined_preds)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─
# PLOTS
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*─
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG  = '#1a1d2e'
TEXT_COL = '#e2e8f0'
GRID_COL = '#2d3148'
COLORS   = ['#4ade80', '#a78bfa', '#f87171']
NAMES    = ['Isolation Forest', 'One-Class SVM', 'Combined (Layered)']

plt.rcParams.update({
    'text.color':       TEXT_COL,
    'axes.labelcolor':  TEXT_COL,
    'xtick.color':      TEXT_COL,
    'ytick.color':      TEXT_COL,
})

# -* Plot 1: Precision / Recall / F1 bar chart -*-*-*-*-*-*─
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(DARK_BG)
x   = np.arange(3)
w   = 0.22
metrics_data = [
    [if_p,  svm_p,  cm_p],
    [if_r,  svm_r,  cm_r],
    [if_f1, svm_f1, cm_f1],
]
metric_names  = ['Precision', 'Recall', 'F1']
metric_colors = ['#4ade80', '#fbbf24', '#a78bfa']
for i, (vals, mname, mc) in enumerate(zip(metrics_data, metric_names, metric_colors)):
    bars = ax1.bar(x + i*w, vals, w, label=mname, color=mc, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7.5,
                 color=TEXT_COL)
ax1.set_xticks(x + w)
ax1.set_xticklabels(['IF', 'SVM', 'Combined'], fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.set_title('Precision / Recall / F1', color=TEXT_COL, fontsize=11, pad=10)
ax1.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)
ax1.spines[:].set_color(GRID_COL)
ax1.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
ax1.set_axisbelow(True)

# -* Plot 2: ROC curves -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(DARK_BG)
for fpr, tpr, auc_val, name, col in [
    (if_fpr,  if_tpr,  if_auc,  'Isolation Forest',  COLORS[0]),
    (svm_fpr, svm_tpr, svm_auc, 'One-Class SVM',     COLORS[1]),
    (cm_fpr,  cm_tpr,  cm_auc,  'Combined',          COLORS[2]),
]:
    ax2.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc_val:.3f})')
ax2.plot([0,1],[0,1], '--', color='#475569', lw=1, label='Random (AUC=0.5)')
ax2.set_xlabel('False Positive Rate', fontsize=9)
ax2.set_ylabel('True Positive Rate',  fontsize=9)
ax2.set_title('ROC Curves', color=TEXT_COL, fontsize=11, pad=10)
ax2.legend(fontsize=7.5, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)
ax2.spines[:].set_color(GRID_COL)
ax2.grid(color=GRID_COL, linewidth=0.5)

# -* Plots 3-5: Confusion matrices -*-*-*-*-*-*-*-*-*-*-*-*
for idx, (cm_data, name) in enumerate([
    (cm_if,   'Isolation Forest'),
    (cm_svm,  'One-Class SVM'),
    (cm_comb, 'Combined (Layered)'),
]):
    row = 1 if idx < 3 else 2
    ax  = fig.add_subplot(gs[1, idx])
    ax.set_facecolor(DARK_BG)
    im  = ax.imshow(cm_data, cmap='Blues', aspect='auto')

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val   = cm_data[i, j]
            total = cm_data[i].sum()
            pct   = val / total * 100 if total > 0 else 0
            ax.text(j, i, f'{val}\n({pct:.1f}%)',
                    ha='center', va='center',
                    color='white' if val > cm_data.max()/2 else TEXT_COL,
                    fontsize=9, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNormal', 'Predicted\nAnomaly'], fontsize=8)
    ax.set_yticklabels(['Actual\nNormal', 'Actual\nAnomaly'], fontsize=8)
    ax.set_title(name, color=TEXT_COL, fontsize=10, pad=8)
    ax.spines[:].set_color(GRID_COL)

# -* Detection rate by difficulty -*-*-*-*-*-*-*-*-*-*-*-*─
# (small inset text summary on the figure)
summary_lines = [
    f"Detection by difficulty (Combined model):",
    f"  Large anomaly  (obvious attacker) : {recall_score(y_true[n:2*n],   combined_preds[n:2*n],   zero_division=0):.1%}",
    f"  Medium anomaly (moderate)         : {recall_score(y_true[2*n:3*n], combined_preds[2*n:3*n], zero_division=0):.1%}",
    f"  Small anomaly  (subtle attacker)  : {recall_score(y_true[3*n:4*n], combined_preds[3*n:4*n], zero_division=0):.1%}",
]
fig.text(0.02, 0.01, "\n".join(summary_lines),
         fontsize=8, color='#94a3b8',
         fontfamily='monospace',
         verticalalignment='bottom')

fig.suptitle(
    'NeuraShield - Zero-Trust Auth Layer  |||  Model Evaluation Report |||',
    fontsize=14, color=TEXT_COL, y=0.98, fontweight='bold'
)

out_path = os.path.join(RESULTS_DIR, "evaluation_report.png")


plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\nSaved: {out_path}")
print("\nDone. Open results/evaluation_report.png to see the full report.")