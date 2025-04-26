import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import io
from PIL import Image

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def visualize_pipeline():
    """Create a comprehensive visualization of the entire antonym detection pipeline"""
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2])
    
    # 1. Input Data Section
    ax1 = fig.add_subplot(gs[0, 0])
    visualize_input_data(ax1)
    
    # 2. Graph Construction Section
    ax2 = fig.add_subplot(gs[0, 1])
    visualize_graph_construction(ax2)
    
    # 3. Model Architecture Section (3D)
    ax3 = fig.add_subplot(gs[0:2, 2], projection='3d')
    visualize_model_architecture(ax3)
    
    # 4. Contrastive Learning Section
    ax4 = fig.add_subplot(gs[1, 0])
    visualize_contrastive_learning(ax4)
    
    # 5. Inference Section
    ax5 = fig.add_subplot(gs[1, 1])
    visualize_inference(ax5)
    
    plt.tight_layout()
    plt.savefig('assets/antonym_detection_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'assets/antonym_detection_pipeline.png'

def visualize_input_data(ax):
    """Visualize the input data preprocessing"""
    ax.axis('off')
    ax.set_title("1. Input Data Preparation", fontsize=14, fontweight='bold')
    
    # Draw sample word pairs
    samples = [
        ("hot", "cold", "Antonym"),
        ("large", "small", "Antonym"),
        ("happy", "joyful", "Not Antonym"),
        ("book", "page", "Not Antonym")
    ]
    
    for i, (word1, word2, label) in enumerate(samples):
        y_pos = 0.8 - i * 0.2
        color = "lightcoral" if label == "Antonym" else "lightblue"
        # Word pair box
        ax.add_patch(plt.Rectangle((0.1, y_pos-0.08), 0.8, 0.15, 
                                  fill=True, color=color, alpha=0.3))
        # Words and label
        ax.text(0.15, y_pos, f"{word1}", fontsize=11)
        ax.text(0.4, y_pos, f"{word2}", fontsize=11)
        ax.text(0.65, y_pos, f"{label}", fontsize=10, style='italic')
    
    # BERT embedding box
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.2, 
                              fill=True, color="lightyellow", alpha=0.3))
    ax.text(0.15, 0.15, "BERT Embedding Layer", fontsize=12, fontweight='bold')
    ax.text(0.15, 0.08, "Converts words to 768-dim vectors", fontsize=10)
    
    # Draw arrows
    ax.arrow(0.5, 0.25, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

def visualize_graph_construction(ax):
    """Visualize the graph construction process"""
    ax.axis('off')
    ax.set_title("2. Graph Construction", fontsize=14, fontweight='bold')
    
    # Create a simple graph representation
    G = nx.DiGraph()
    pos = {1: (0.3, 0.7), 2: (0.7, 0.7)}
    G.add_nodes_from([1, 2])
    G.add_edges_from([(1, 2), (2, 1)])
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Add node labels
    ax.text(0.3, 0.7, "Word 1\nEmb", ha='center', va='center', fontsize=11)
    ax.text(0.7, 0.7, "Word 2\nEmb", ha='center', va='center', fontsize=11)
    
    # Add explanation text
    ax.text(0.5, 0.4, "Graph Structure:", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.35, "• Two nodes with BERT embeddings as features", ha='center', fontsize=10)
    ax.text(0.5, 0.3, "• Bidirectional edges between word nodes", ha='center', fontsize=10)
    ax.text(0.5, 0.25, "• Label indicates antonym relationship", ha='center', fontsize=10)
    
    # Add mathematical representation
    ax.add_patch(plt.Rectangle((0.2, 0.1), 0.6, 0.1, fill=True, color="lavender", alpha=0.3))
    ax.text(0.5, 0.15, "x = torch.stack([emb1, emb2])", ha='center', fontsize=10)
    ax.text(0.5, 0.1, "edge_index = tensor([[0, 1], [1, 0]])", ha='center', fontsize=10)
    ax.text(0.5, 0.05, "y = tensor(label)", ha='center', fontsize=10)

def visualize_model_architecture(ax):
    """Visualize the hierarchical GAT model architecture in 3D"""
    ax.set_title("3. Hierarchical GAT with Contrastive Learning", fontsize=14, fontweight='bold')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # Input layer
    x1, y1 = [0.2, 0.8], [0.1, 0.1]
    z1 = [0.9, 0.9]
    ax.scatter(x1, y1, z1, s=100, c='skyblue', alpha=0.8)
    ax.text(0.2, 0.1, 0.95, "Node 1\n(Word 1)", fontsize=9, ha='center')
    ax.text(0.8, 0.1, 0.95, "Node 2\n(Word 2)", fontsize=9, ha='center')
    
    # Draw edge between input nodes
    arrow1 = Arrow3D([0.2, 0.8], [0.1, 0.1], [0.9, 0.9], 
                    mutation_scale=15, lw=2, arrowstyle='->', color='gray')
    ax.add_artist(arrow1)
    arrow2 = Arrow3D([0.8, 0.2], [0.1, 0.1], [0.9, 0.9], 
                    mutation_scale=15, lw=2, arrowstyle='->', color='gray')
    ax.add_artist(arrow2)
    
    # GAT Layer 1
    ax.text(0.5, 0.05, 0.75, "GAT Layer 1 (4 heads)", fontsize=10, ha='center')
    x2, y2 = [0.3, 0.7], [0.2, 0.2]
    z2 = [0.7, 0.7]
    ax.scatter(x2, y2, z2, s=120, c='lightgreen', alpha=0.8)
    
    # Draw arrows from input to GAT1
    for xi, yi, zi in zip(x1, y1, z1):
        for xj, yj, zj in zip(x2, y2, z2):
            arrow = Arrow3D([xi, xj], [yi, yj], [zi, zj], 
                          mutation_scale=15, lw=1, arrowstyle='->', color='gray', alpha=0.4)
            ax.add_artist(arrow)
    
    # SAGPooling
    ax.text(0.5, 0.05, 0.55, "SAGPooling (ratio=0.5)", fontsize=10, ha='center')
    
    # GAT Layer 2
    ax.text(0.5, 0.05, 0.4, "GAT Layer 2 (1 head)", fontsize=10, ha='center')
    x3, y3 = [0.4, 0.6], [0.3, 0.3]
    z3 = [0.4, 0.4]
    ax.scatter(x3, y3, z3, s=120, c='lightsalmon', alpha=0.8)
    
    # Draw arrows from GAT1 to GAT2
    for xi, yi, zi in zip(x2, y2, z2):
        for xj, yj, zj in zip(x3, y3, z3):
            arrow = Arrow3D([xi, xj], [yi, yj], [zi, zj], 
                          mutation_scale=15, lw=1, arrowstyle='->', color='gray', alpha=0.4)
            ax.add_artist(arrow)
    
    # Global pooling
    ax.text(0.5, 0.05, 0.25, "Global Mean Pooling", fontsize=10, ha='center')
    x4, y4, z4 = 0.5, 0.4, 0.2
    ax.scatter(x4, y4, z4, s=150, c='plum', alpha=0.8)
    
    # Draw arrows to global pooling
    for xi, yi, zi in zip(x3, y3, z3):
        arrow = Arrow3D([xi, x4], [yi, y4], [zi, z4], 
                      mutation_scale=15, lw=1, arrowstyle='->', color='gray', alpha=0.4)
        ax.add_artist(arrow)
    
    # Output layers
    ax.text(0.3, 0.5, 0.1, "Classification\nHead", fontsize=9, ha='center')
    ax.text(0.7, 0.5, 0.1, "Projection\nHead", fontsize=9, ha='center')
    
    x5, y5 = [0.3, 0.7], [0.5, 0.5]
    z5 = [0.1, 0.1]
    ax.scatter(x5, y5, z5, s=150, c=['gold', 'orchid'], alpha=0.8)
    
    # Draw arrows to output layers
    arrow1 = Arrow3D([x4, x5[0]], [y4, y5[0]], [z4, z5[0]], 
                    mutation_scale=15, lw=1, arrowstyle='->', color='gray')
    ax.add_artist(arrow1)
    arrow2 = Arrow3D([x4, x5[1]], [y4, y5[1]], [z4, z5[1]], 
                    mutation_scale=15, lw=1, arrowstyle='->', color='gray')
    ax.add_artist(arrow2)
    
    # Add model parameters notation
    ax.text(0.1, 0.8, 0.3, "Model Parameters:", fontsize=10, ha='left')
    ax.text(0.1, 0.8, 0.25, "• Input: 768-dim (BERT)", fontsize=9, ha='left')
    ax.text(0.1, 0.8, 0.2, "• Hidden: 256-dim", fontsize=9, ha='left')
    ax.text(0.1, 0.8, 0.15, "• Output: 2 classes", fontsize=9, ha='left')
    ax.text(0.1, 0.8, 0.1, "• Projection: 128-dim", fontsize=9, ha='left')

def visualize_contrastive_learning(ax):
    """Visualize the contrastive learning approach"""
    ax.axis('off')
    ax.set_title("4. Contrastive Learning Approach", fontsize=14, fontweight='bold')
    
    # Create embedding space visualization
    # Randomly generate points for antonyms and non-antonyms
    np.random.seed(42)
    
    # Simulate embeddings in 2D space
    antonym_pairs = np.random.randn(6, 2) * 0.2 + np.array([0.3, 0.6])
    non_antonym_pairs = np.random.randn(6, 2) * 0.2 + np.array([0.7, 0.6])
    
    # Plot points
    ax.scatter(antonym_pairs[:, 0], antonym_pairs[:, 1], s=80, c='lightcoral', 
              label='Antonym Pairs', alpha=0.8)
    ax.scatter(non_antonym_pairs[:, 0], non_antonym_pairs[:, 1], s=80, c='lightblue', 
               label='Non-Antonym Pairs', alpha=0.8)
    
    # Draw some positive and negative pair connections
    # Positive pairs (same class)
    for i in range(2):
        ax.plot([antonym_pairs[i, 0], antonym_pairs[i+2, 0]],
                [antonym_pairs[i, 1], antonym_pairs[i+2, 1]],
                'g-', alpha=0.6, linewidth=1.5)
        ax.plot([non_antonym_pairs[i, 0], non_antonym_pairs[i+2, 0]],
                [non_antonym_pairs[i, 1], non_antonym_pairs[i+2, 1]],
                'g-', alpha=0.6, linewidth=1.5)
    
    # Negative pairs (different classes)
    for i in range(2):
        ax.plot([antonym_pairs[i, 0], non_antonym_pairs[i, 0]],
                [antonym_pairs[i, 1], non_antonym_pairs[i, 1]],
                'r--', alpha=0.6, linewidth=1.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9)
    
    # Add explanation text
    ax.text(0.05, 0.25, "Contrastive Loss Components:", fontsize=11, fontweight='bold')
    ax.text(0.07, 0.2, "• Pull similar pairs closer (same class)", fontsize=9)
    ax.text(0.07, 0.15, "• Push dissimilar pairs apart (different classes)", fontsize=9)
    ax.text(0.07, 0.1, "• Temperature parameter: 0.07", fontsize=9)
    ax.text(0.07, 0.05, "• Projection dimension: 128", fontsize=9)

def visualize_inference(ax):
    """Visualize the inference process"""
    ax.axis('off')
    ax.set_title("5. Inference Process", fontsize=14, fontweight='bold')
    
    # Create a flow diagram for inference
    stages = [
        "New Word Pair\n('large', 'huge')",
        "Convert to Graph\nRepresentation",
        "Process through\nHierarchical GAT",
        "Apply Test-Time\nAugmentation",
        "Final Prediction\n('Not Antonym')"
    ]
    
    y_pos = 0.8
    for i, stage in enumerate(stages):
        # Draw box
        color = 'lightblue' if i % 2 == 0 else 'lightyellow'
        ax.add_patch(plt.Rectangle((0.1, y_pos-0.08), 0.8, 0.15, 
                                  fill=True, color=color, alpha=0.3))
        # Add text
        ax.text(0.5, y_pos, stage, ha='center', fontsize=10)
        
        # Add arrow
        if i < len(stages) - 1:
            ax.arrow(0.5, y_pos-0.08, 0, -0.05, head_width=0.02, 
                    head_length=0.02, fc='black', ec='black')
        
        y_pos -= 0.2
    
    # Add detail about test-time augmentation
    ax.add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.15, 
                              fill=True, color='lavender', alpha=0.3))
    ax.text(0.5, 0.13, "Test-Time Augmentation:", ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.08, "Multiple forward passes with dropout enabled", ha='center', fontsize=9)
    ax.text(0.5, 0.05, "Average predictions to improve robustness", ha='center', fontsize=9)

if __name__ == "__main__":
    # Create the pipeline visualization
    output_path = visualize_pipeline()
    print(f"Pipeline visualization saved to {output_path}")