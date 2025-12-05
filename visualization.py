"""
Create visualizations for RAG evaluation results
Save charts to outputs/figures/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
df = pd.read_csv('ragas_evaluation_results.csv')

# Create outputs/figures directory if it doesn't exist
import os
os.makedirs('outputs/figures', exist_ok=True)

# ============================================================================
# 1. Overall Metrics Bar Chart
# ============================================================================

metrics = ['context_recall', 'context_precision', 'answer_relevancy', 
           'faithfulness', 'answer_correctness']
scores = [df[metric].mean() for metric in metrics]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}',
             ha='center', va='bottom', fontweight='bold')

# Add threshold line at 80%
plt.axhline(y=0.80, color='green', linestyle='--', alpha=0.7, label='80% Target')

plt.title('MedBot RAG Evaluation Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.0)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/metrics_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/metrics_overview.png")
plt.close()

# ============================================================================
# 2. Metrics Distribution Box Plot
# ============================================================================

plt.figure(figsize=(12, 6))
df[metrics].boxplot(vert=True, patch_artist=True)
plt.title('Distribution of Evaluation Metrics Across 129 Questions', 
          fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/metrics_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/metrics_distribution.png")
plt.close()

# ============================================================================
# 3. Correlation Heatmap
# ============================================================================

plt.figure(figsize=(10, 8))
correlation = df[metrics].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Between Evaluation Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/metrics_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/metrics_correlation.png")
plt.close()

# ============================================================================
# 4. Performance Distribution Histogram
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].hist(df[metric], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].axvline(df[metric].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {df[metric].mean():.2%}')
    axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
    axes[idx].set_xlabel('Score')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# Remove empty subplot
axes[-1].remove()

plt.suptitle('Score Distribution for Each Metric', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('outputs/figures/score_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/score_distributions.png")
plt.close()

# ============================================================================
# 5. Top 10 Best vs Worst Performing Questions
# ============================================================================

# Calculate average score across all metrics
df['avg_score'] = df[metrics].mean(axis=1)

# Get top and bottom 10
top_10 = df.nlargest(10, 'avg_score')
bottom_10 = df.nsmallest(10, 'avg_score')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 10
top_10_sorted = top_10.sort_values('avg_score')
ax1.barh(range(len(top_10_sorted)), top_10_sorted['avg_score'], color='green', alpha=0.7)
ax1.set_yticks(range(len(top_10_sorted)))
ax1.set_yticklabels([f"Q{i}" for i in range(1, 11)], fontsize=8)
ax1.set_xlabel('Average Score')
ax1.set_title('Top 10 Best Performing Questions', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Bottom 10
bottom_10_sorted = bottom_10.sort_values('avg_score')
ax2.barh(range(len(bottom_10_sorted)), bottom_10_sorted['avg_score'], color='red', alpha=0.7)
ax2.set_yticks(range(len(bottom_10_sorted)))
ax2.set_yticklabels([f"Q{i}" for i in range(1, 11)], fontsize=8)
ax2.set_xlabel('Average Score')
ax2.set_title('Top 10 Worst Performing Questions', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.suptitle('Performance Extremes Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/performance_extremes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/performance_extremes.png")
plt.close()

# ============================================================================
# 6. Radar Chart for Overall Performance
# ============================================================================

from math import pi

categories = [m.replace('_', '\n').title() for m in metrics]
values = scores + [scores[0]]  # Close the circle

angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='MedBot')
ax.fill(angles, values, alpha=0.25, color='blue')

# Add benchmark line at 80%
benchmark = [0.80] * len(values)
ax.plot(angles, benchmark, '--', linewidth=2, color='green', label='80% Target', alpha=0.7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.title('MedBot Performance Radar Chart', size=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/performance_radar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/performance_radar.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for metric in metrics:
    mean = df[metric].mean()
    std = df[metric].std()
    min_val = df[metric].min()
    max_val = df[metric].max()
    print(f"\n{metric}:")
    print(f"  Mean: {mean:.3f} ({mean*100:.1f}%)")
    print(f"  Std:  {std:.3f}")
    print(f"  Min:  {min_val:.3f}")
    print(f"  Max:  {max_val:.3f}")

print("\n" + "="*60)
print("✓ All visualizations saved to outputs/figures/")
print("="*60)