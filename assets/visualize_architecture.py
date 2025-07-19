#!/usr/bin/env python3
"""
Generate SVG illustrations for Bhav-Net paper
Creates architectural diagrams and analysis visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

def create_dual_encoder_architecture():
    """Create the main architecture diagram for Bhav-Net"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    bert_color = '#FF6B6B'
    projection_color = '#4ECDC4' 
    graph_color = '#45B7D1'
    classifier_color = '#96CEB4'
    
    # Input words
    ax.text(0.5, 7, 'Word Pair\n(w₁, w₂)', ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Language-specific BERT encoders
    bert_box1 = FancyBboxPatch((0.2, 5.5), 1.6, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=bert_color, alpha=0.7, linewidth=2)
    ax.add_patch(bert_box1)
    ax.text(1, 5.9, 'Language-Specific\nBERT Encoder', ha='center', va='center', fontsize=10, weight='bold')
    
    # Dual projection branches
    # Synonym branch
    syn_box = FancyBboxPatch((2.5, 6.2), 2, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=projection_color, alpha=0.7, linewidth=2)
    ax.add_patch(syn_box)
    ax.text(3.5, 6.5, 'Synonym Projection\nW_syn · x + b_syn', ha='center', va='center', fontsize=9, weight='bold')
    
    # Antonym branch  
    ant_box = FancyBboxPatch((2.5, 5.2), 2, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=projection_color, alpha=0.7, linewidth=2)
    ax.add_patch(ant_box)
    ax.text(3.5, 5.5, 'Antonym Projection\nW_ant · x + b_ant', ha='center', va='center', fontsize=9, weight='bold')
    
    # Graph construction
    graph_box = FancyBboxPatch((5.5, 5.5), 1.8, 1.2, boxstyle="round,pad=0.1", 
                              facecolor=graph_color, alpha=0.7, linewidth=2)
    ax.add_patch(graph_box)
    ax.text(6.4, 6.1, 'Graph\nConstruction\n(2-node graph)', ha='center', va='center', fontsize=9, weight='bold')
    
    # Graph transformer
    transformer_box = FancyBboxPatch((5.5, 3.8), 1.8, 1.2, boxstyle="round,pad=0.1", 
                                    facecolor=graph_color, alpha=0.7, linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(6.4, 4.4, 'Graph\nTransformer\nConvolution', ha='center', va='center', fontsize=9, weight='bold')
    
    # Classification head
    classifier_box = FancyBboxPatch((8, 4.5), 1.5, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor=classifier_color, alpha=0.7, linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(8.75, 5.1, 'Classification\nHead\n+ Margin Loss', ha='center', va='center', fontsize=9, weight='bold')
    
    # Output
    ax.text(9.5, 2.5, 'Antonym\nPrediction\n(0 or 1)', ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to BERT
    ax.annotate('', xy=(1, 6.3), xytext=(1, 7), arrowprops=arrow_props)
    
    # BERT to projections
    ax.annotate('', xy=(2.5, 6.5), xytext=(1.8, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 5.5), xytext=(1.8, 5.8), arrowprops=arrow_props)
    
    # Projections to graph
    ax.annotate('', xy=(5.5, 6.2), xytext=(4.5, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 5.8), xytext=(4.5, 5.5), arrowprops=arrow_props)
    
    # Graph construction to transformer
    ax.annotate('', xy=(6.4, 5.0), xytext=(6.4, 5.5), arrowprops=arrow_props)
    
    # Transformer to classifier
    ax.annotate('', xy=(8.0, 5.1), xytext=(7.3, 4.4), arrowprops=arrow_props)
    
    # Classifier to output
    ax.annotate('', xy=(9.2, 3.2), xytext=(8.9, 4.5), arrowprops=arrow_props)
    
    # Add semantic space illustrations
    ax.text(3.5, 7.2, 'Dual Semantic Spaces', ha='center', va='center', fontsize=12, weight='bold')
    
    # Add mathematical formulations
    ax.text(1, 4.8, 'x_w ∈ ℝ⁷⁶⁸', ha='center', va='center', fontsize=8, style='italic')
    ax.text(3.5, 4.8, 'x_syn, x_ant ∈ ℝʰ', ha='center', va='center', fontsize=8, style='italic')
    
    plt.title('Bhav-Net: Dual-Space Graph Transformer Architecture', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/threesamyak/hsl844/Antonym-detection/assets/dual_encoder_architecture.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated dual_encoder_architecture.svg")

def create_semantic_space_analysis():
    """Create visualization showing how dual spaces organize semantic relationships"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate sample data for visualization
    np.random.seed(42)
    
    # Synonym space
    # Synonyms cluster together
    syn_clusters = []
    cluster_centers = [(2, 2), (7, 7), (3, 8), (8, 3)]
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (cx, cy) in enumerate(cluster_centers):
        cluster = np.random.normal([cx, cy], 0.8, (8, 2))
        syn_clusters.append((cluster, colors[i]))
    
    # Antonyms scattered (dissimilar in synonym space)
    antonym_pairs = [
        (np.array([2.5, 2.5]), np.array([7.5, 7.5])),  # hot-cold
        (np.array([3, 8]), np.array([8, 3])),          # big-small
        (np.array([1, 6]), np.array([9, 4])),          # happy-sad
    ]
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Synonym Space\n(Synonyms cluster, Antonyms scatter)', fontsize=12, weight='bold')
    
    # Plot synonym clusters
    for cluster, color in syn_clusters:
        ax1.scatter(cluster[:, 0], cluster[:, 1], c=color, alpha=0.6, s=50, label=f'Synonym cluster')
    
    # Plot antonym pairs with lines
    for i, (ant1, ant2) in enumerate(antonym_pairs):
        ax1.scatter(ant1[0], ant1[1], c='black', s=100, marker='s', alpha=0.8)
        ax1.scatter(ant2[0], ant2[1], c='black', s=100, marker='s', alpha=0.8)
        ax1.plot([ant1[0], ant2[0]], [ant1[1], ant2[1]], 'k--', alpha=0.5, linewidth=2)
        if i == 0:
            ax1.text((ant1[0]+ant2[0])/2, (ant1[1]+ant2[1])/2 + 0.3, 'hot-cold', 
                    ha='center', fontsize=9, weight='bold')
    
    ax1.set_xlabel('Dimension 1', fontsize=10)
    ax1.set_ylabel('Dimension 2', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Antonym space
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Antonym Space\n(Antonyms cluster, Synonyms scatter)', fontsize=12, weight='bold')
    
    # In antonym space, antonym pairs cluster together
    antonym_cluster_center = (5, 5)
    antonym_cluster = []
    for ant1, ant2 in antonym_pairs:
        # Move pairs close together in antonym space
        center = np.array(antonym_cluster_center) + np.random.normal(0, 1.5, 2)
        pair_points = np.random.normal(center, 0.3, (2, 2))
        antonym_cluster.extend(pair_points)
        ax2.scatter(pair_points[:, 0], pair_points[:, 1], c='purple', s=100, marker='s', alpha=0.8)
        ax2.plot(pair_points[:, 0], pair_points[:, 1], 'purple', linewidth=3, alpha=0.7)
    
    # Synonyms are now scattered
    for cluster, color in syn_clusters:
        # Scatter the synonym clusters in antonym space
        scattered = cluster + np.random.normal(0, 2, cluster.shape)
        ax2.scatter(scattered[:, 0], scattered[:, 1], c=color, alpha=0.4, s=30)
    
    ax2.set_xlabel('Dimension 1', fontsize=10)
    ax2.set_ylabel('Dimension 2', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend(['Synonyms'], loc='upper right')
    ax2.legend(['Antonym pairs'], loc='upper right')
    
    plt.suptitle('Dual-Space Semantic Organization in Bhav-Net', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('/home/threesamyak/hsl844/Antonym-detection/assets/semantic_space_analysis.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated semantic_space_analysis.svg")

def create_training_pipeline():
    """Create visualization of the two-stage training process"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Stage 1: BERT Fine-tuning
    stage1_box = FancyBboxPatch((0.5, 3.5), 4.5, 2, boxstyle="round,pad=0.2", 
                               facecolor='lightblue', alpha=0.7, linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(2.75, 4.5, 'Stage 1: BERT Fine-tuning\n(3-5 epochs)', ha='center', va='center', 
            fontsize=14, weight='bold')
    ax.text(2.75, 3.8, '• Language-specific optimization\n• Binary classification warm-up\n• High-quality embeddings', 
            ha='center', va='center', fontsize=10)
    
    # Stage 2: Dual Encoder Training
    stage2_box = FancyBboxPatch((6.5, 3.5), 4.5, 2, boxstyle="round,pad=0.2", 
                               facecolor='lightgreen', alpha=0.7, linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(8.75, 4.5, 'Stage 2: Dual Encoder Training\n(10-20 epochs)', ha='center', va='center', 
            fontsize=14, weight='bold')
    ax.text(8.75, 3.8, '• Frozen BERT encoders\n• Dual-space projection learning\n• Margin-based optimization', 
            ha='center', va='center', fontsize=10)
    
    # Arrow between stages
    arrow_props = dict(arrowstyle='->', lw=3, color='darkred')
    ax.annotate('', xy=(6.5, 4.5), xytext=(5.0, 4.5), arrowprops=arrow_props)
    
    # Loss functions
    ax.text(2.75, 2.8, 'ℒ_BERT = BCE Loss', ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(8.75, 2.8, 'ℒ = ℒ_BCE + λ·ℒ_margin', ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Benefits
    ax.text(2.75, 1.8, '✓ Language-specific patterns\n✓ Stable initialization', ha='center', va='center', 
            fontsize=9, color='blue')
    ax.text(8.75, 1.8, '✓ Dual-space organization\n✓ Semantic relationship modeling', ha='center', va='center', 
            fontsize=9, color='green')
    
    plt.title('Two-Stage Training Strategy for Bhav-Net', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/threesamyak/hsl844/Antonym-detection/assets/training_pipeline.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated training_pipeline.svg")

def create_performance_comparison():
    """Create performance comparison chart"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Data from the paper
    methods = ['ICENET\nBaseline', 'Distiller\nBaseline', 'Bhav-Net\n(Dual Encoder)']
    adjectives = [0.82, 0.80, 0.90]
    verbs = [0.85, 0.83, 0.93]
    nouns = [0.79, 0.77, 0.90]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, adjectives, width, label='Adjectives', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x, verbs, width, label='Verbs', color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + width, nouns, width, label='Nouns', color='#45B7D1', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_ylabel('F1-Score', fontsize=12, weight='bold')
    ax.set_xlabel('Method', fontsize=12, weight='bold')
    ax.set_title('Performance Comparison on English Benchmarks', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight Bhav-Net improvements
    ax.axvline(x=2, color='gold', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(2, 0.95, 'Bhav-Net\nImprovements', ha='center', va='center', fontsize=10, 
            weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/threesamyak/hsl844/Antonym-detection/assets/performance_comparison.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated performance_comparison.svg")

def create_cross_lingual_analysis():
    """Create cross-lingual effectiveness visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Language performance chart
    languages = ['German', 'French', 'Spanish', 'Italian', 'Portuguese', 'Dutch', 'Russian']
    f1_scores = [0.85, 0.88, 0.86, 0.84, 0.87, 0.83, 0.82]
    dataset_sizes = [2678, 6095, 2263, 2495, 2257, 1865, 1279]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(languages)))
    
    bars = ax1.bar(languages, f1_scores, color=colors, alpha=0.8)
    ax1.set_ylabel('F1-Score', fontsize=12, weight='bold')
    ax1.set_xlabel('Language', fontsize=12, weight='bold')
    ax1.set_title('Cross-Lingual Performance', fontsize=14, weight='bold')
    ax1.set_ylim(0.75, 0.90)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Dataset size vs performance scatter
    ax2.scatter(dataset_sizes, f1_scores, c=colors, s=100, alpha=0.8)
    
    for i, lang in enumerate(languages):
        ax2.annotate(lang, (dataset_sizes[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Trend line
    z = np.polyfit(dataset_sizes, f1_scores, 1)
    p = np.poly1d(z)
    ax2.plot(dataset_sizes, p(dataset_sizes), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Dataset Size (word pairs)', fontsize=12, weight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, weight='bold')
    ax2.set_title('Performance vs Dataset Size', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Lingual Effectiveness of Bhav-Net', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('/home/threesamyak/hsl844/Antonym-detection/assets/cross_lingual_analysis.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated cross_lingual_analysis.svg")

if __name__ == "__main__":
    print("Generating SVG illustrations for Bhav-Net paper...")
    
    create_dual_encoder_architecture()
    create_semantic_space_analysis()
    create_training_pipeline()
    create_performance_comparison()
    create_cross_lingual_analysis()
    
    print("\nAll SVG illustrations generated successfully!")
    print("Files created:")
    print("- dual_encoder_architecture.svg")
    print("- semantic_space_analysis.svg") 
    print("- training_pipeline.svg")
    print("- performance_comparison.svg")
    print("- cross_lingual_analysis.svg")
