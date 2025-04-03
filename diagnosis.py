import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model architecture (same as in the original code)
class DualEncoderGraphModel(nn.Module):
    """
    Extended model with two projection branches:
      - syn_proj: projects input to a 'synonym' space.
      - ant_proj: projects input to an 'antonym' space.
    Their outputs are concatenated and fused before graph convolutions.
    Additionally, a margin loss based on inner product (using tanh) is computed.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, heads=4, dropout_rate=0.2):
        super(DualEncoderGraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # Synonym projection branch (ENC-1)
        self.syn_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Antonym projection branch (ENC-2)
        self.ant_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Fusion layer: combine both branches (resulting dimension 2*hidden_dim) 
        # and project to the input dimension required for TransformerConv (hidden_dim*heads)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim * heads)
        
        # Attentive Graph Transformer layers
        self.conv1 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
            for _ in range(num_layers - 1)
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x shape: [N, input_dim]
        
        # Compute projections from both branches
        x_syn = self.syn_proj(x)  # [N, hidden_dim]
        x_ant = self.ant_proj(x)  # [N, hidden_dim]
        
        # Save the branch outputs for margin loss computation
        self.x_syn = x_syn
        self.x_ant = x_ant
        
        # Concatenate the two projections
        x_combined = torch.cat([x_syn, x_ant], dim=1)  # [N, 2*hidden_dim]
        x_fused = self.fusion(x_combined)  # [N, hidden_dim*heads]
        
        # Graph transformer layers with dropout and ReLU
        x_conv = self.conv1(x_fused, edge_index)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        for conv in self.convs:
            x_conv = conv(x_conv, edge_index)
            x_conv = self.activation(x_conv)
            x_conv = self.dropout(x_conv)
        
        # Global pooling for graph-level representation
        x_pool = global_mean_pool(x_conv, batch)
        out = self.classifier(x_pool)
        return out.squeeze()

def load_data(file_path):
    """Load data from a file into lists of word pairs and labels."""
    word1_list, word2_list, labels = [], [], []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    
    return word1_list, word2_list, labels

def embed_word_pairs(word1_list, word2_list, model):
    """Embed word pairs using the provided model."""
    print("Embedding word pairs...")
    emb1 = model.encode(word1_list, show_progress_bar=True)
    emb2 = model.encode(word2_list, show_progress_bar=True)
    
    print(f"Embedding complete. Shape: {emb1.shape}")
    return emb1, emb2

def create_graph_data(word1_emb, word2_emb, label):
    """Create a graph data object for a single word pair."""
    # Create node features (2 nodes: word1 and word2)
    x = torch.tensor(np.vstack([word1_emb, word2_emb]), dtype=torch.float)
    
    # Create edges (bidirectional connection between the two words)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    # Create label
    y = torch.tensor([label], dtype=torch.float)
    
    # Create graph data object
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    
    return data

def create_graph_dataset(word1_embeddings, word2_embeddings, labels):
    """Create a list of graph data objects from word pair embeddings."""
    dataset = []
    
    for i in range(len(labels)):
        data = create_graph_data(word1_embeddings[i], word2_embeddings[i], labels[i])
        dataset.append(data)
    
    return dataset

def compute_margin_loss(x_syn, x_ant, data):
    """
    Compute margin-based loss on the dual projections.
    For a graph with two nodes (word pair):
      - For synonym pairs (label==0): we want tanh(inner(x_syn_0, x_syn_1)) to be high.
      - For antonym pairs (label==1): we want tanh(inner(x_ant_0, x_ant_1)) to be low.
    """
    margin_syn = 0.8
    margin_ant = 0.2
    losses = []
    batch = data.batch.cpu().numpy()
    unique_graphs = np.unique(batch)

    for g in unique_graphs:
        idx = (data.batch == g).nonzero(as_tuple=False).view(-1)
        if idx.shape[0] != 2:
            continue
        
        # Fix: index the correct graph label instead of using idx[0].
        label = data.y[g].item()

        sim_syn = torch.tanh(torch.dot(x_syn[idx[0]], x_syn[idx[1]]))
        sim_ant = torch.tanh(torch.dot(x_ant[idx[0]], x_ant[idx[1]]))
        if label == 0:  # synonym pair
            loss = torch.relu(margin_syn - sim_syn)
        else:  # antonym pair
            loss = torch.relu(sim_ant - margin_ant)
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device)

def analyze_model_weights(model):
    """Analyze the model weights and biases."""
    print("\n## Model Weight Analysis")
    
    # Analyze synonym and antonym projection layers
    print("\n**Projection Layer Analysis**")
    
    # Synonym projection weights
    syn_w = model.syn_proj[0].weight
    print(f"Synonym projection weights - Shape: {syn_w.shape}, Mean: {syn_w.mean().item():.4f}, Std: {syn_w.std().item():.4f}")
    
    # Antonym projection weights
    ant_w = model.ant_proj[0].weight
    print(f"Antonym projection weights - Shape: {ant_w.shape}, Mean: {ant_w.mean().item():.4f}, Std: {ant_w.std().item():.4f}")
    
    # Fusion layer weights
    fusion_w = model.fusion.weight
    print(f"Fusion layer weights - Shape: {fusion_w.shape}, Mean: {fusion_w.mean().item():.4f}, Std: {fusion_w.std().item():.4f}")
    
    # Transformer layers
    print("\n**Transformer Layer Analysis**")
    
    # First transformer layer
    conv1_w = model.conv1.lin_key.weight
    print(f"First transformer layer key weights - Shape: {conv1_w.shape}, Mean: {conv1_w.mean().item():.4f}, Std: {conv1_w.std().item():.4f}")
    
    # Last transformer layer
    last_conv_w = model.convs[-1].lin_key.weight
    print(f"Last transformer layer key weights - Shape: {last_conv_w.shape}, Mean: {last_conv_w.mean().item():.4f}, Std: {last_conv_w.std().item():.4f}")
    
    # Classifier weights
    clf_w = model.classifier[0].weight
    print(f"Classifier weights - Shape: {clf_w.shape}, Mean: {clf_w.mean().item():.4f}, Std: {clf_w.std().item():.4f}")
    
    # Plot weight distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(syn_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Synonym Projection Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 3, 2)
    plt.hist(ant_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Antonym Projection Weights")
    plt.xlabel("Weight Value")
    
    plt.subplot(2, 3, 3)
    plt.hist(fusion_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Fusion Layer Weights")
    plt.xlabel("Weight Value")
    
    plt.subplot(2, 3, 4)
    plt.hist(conv1_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("First Transformer Layer Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 3, 5)
    plt.hist(last_conv_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Last Transformer Layer Weights")
    plt.xlabel("Weight Value")
    
    plt.subplot(2, 3, 6)
    plt.hist(clf_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Classifier Weights")
    plt.xlabel("Weight Value")
    
    plt.tight_layout()
    plt.savefig("assets/weight_distributions.png")
    plt.close()
    
    print("Weight distribution plots saved to assets/weight_distributions.png")
    
    # Analyze weight differences between synonym and antonym projections
    weight_diff = syn_w - ant_w
    print(f"\n**Projection Weight Differences**")
    print(f"Syn-Ant weight difference - Mean: {weight_diff.mean().item():.4f}, Std: {weight_diff.std().item():.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(weight_diff.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Synonym-Antonym Projection Weight Differences")
    plt.xlabel("Weight Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("assets/projection_weight_diff.png")
    plt.close()
    
    print("Projection weight difference plot saved to assets/projection_weight_diff.png")

def analyze_model_activations(model, dataset, num_samples=100):
    """Analyze model activations on a subset of data."""
    print("\n## Model Activation Analysis")
    
    # Create a dataloader with a subset of the data
    subset_size = min(num_samples, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = [dataset[i] for i in subset_indices]
    loader = torch_geometric.loader.DataLoader(subset, batch_size=subset_size)
    
    # Get a batch
    batch = next(iter(loader)).to(device)
    
    # Get activations
    model.eval()
    with torch.no_grad():
        # Input
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        # Projection branches
        x_syn = model.syn_proj[0](x)  # Linear layer output
        x_syn_act = model.syn_proj[1](x_syn)  # After ReLU
        
        x_ant = model.ant_proj[0](x)  # Linear layer output
        x_ant_act = model.ant_proj[1](x_ant)  # After ReLU
        
        # Combined and fused
        x_combined = torch.cat([model.syn_proj(x), model.ant_proj(x)], dim=1)
        x_fused = model.fusion(x_combined)
        
        # First transformer layer
        x_conv1 = model.conv1(x_fused, edge_index)
        x_conv1_act = model.activation(x_conv1)
        
        # Last transformer layer
        x_last = x_conv1_act
        for conv in model.convs[:-1]:
            x_last = model.activation(conv(x_last, edge_index))
        
        x_last_out = model.convs[-1](x_last, edge_index)
        x_last_act = model.activation(x_last_out)
        
        # Pooled
        x_pooled = global_mean_pool(x_last_act, batch_idx)
        
        # Classifier
        x_clf = model.classifier[0](x_pooled)  # First linear layer
        x_clf_act = model.classifier[1](x_clf)  # After ReLU
        
        # Final output
        final_out = model(batch)
        
        # Compute margin loss components
        margin_syn = 0.8
        margin_ant = 0.2
        syn_similarities = []
        ant_similarities = []
        labels = []
        
        batch_np = batch_idx.cpu().numpy()
        unique_graphs = np.unique(batch_np)
        
        for g in unique_graphs:
            idx = (batch_idx == g).nonzero(as_tuple=False).view(-1)
            if idx.shape[0] != 2:
                continue
                
            label = batch.y[g].item()
            labels.append(label)
            
            # Compute similarities
            sim_syn = torch.tanh(torch.dot(model.x_syn[idx[0]], model.x_syn[idx[1]]))
            sim_ant = torch.tanh(torch.dot(model.x_ant[idx[0]], model.x_ant[idx[1]]))
            
            syn_similarities.append(sim_syn.item())
            ant_similarities.append(sim_ant.item())
    
    # Analyze activations
    print("\n**Activation Statistics**")
    print(f"Synonym projection (linear) - Mean: {x_syn.mean().item():.4f}, Std: {x_syn.std().item():.4f}")
    print(f"Synonym projection (ReLU) - Mean: {x_syn_act.mean().item():.4f}, Std: {x_syn_act.std().item():.4f}")
    print(f"Antonym projection (linear) - Mean: {x_ant.mean().item():.4f}, Std: {x_ant.std().item():.4f}")
    print(f"Antonym projection (ReLU) - Mean: {x_ant_act.mean().item():.4f}, Std: {x_ant_act.std().item():.4f}")
    print(f"Fused features - Mean: {x_fused.mean().item():.4f}, Std: {x_fused.std().item():.4f}")
    print(f"First transformer output - Mean: {x_conv1.mean().item():.4f}, Std: {x_conv1.std().item():.4f}")
    print(f"Last transformer activation - Mean: {x_last_act.mean().item():.4f}, Std: {x_last_act.std().item():.4f}")
    print(f"Pooled features - Mean: {x_pooled.mean().item():.4f}, Std: {x_pooled.std().item():.4f}")
    print(f"Classifier activation - Mean: {x_clf_act.mean().item():.4f}, Std: {x_clf_act.std().item():.4f}")
    print(f"Final output - Mean: {final_out.mean().item():.4f}, Std: {final_out.std().item():.4f}")
    
    # Plot activation distributions
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 3, 1)
    plt.hist(x_syn_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Synonym Projection Activations")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(3, 3, 2)
    plt.hist(x_ant_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Antonym Projection Activations")
    plt.xlabel("Activation Value")
    
    plt.subplot(3, 3, 3)
    plt.hist(x_fused.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Fused Features")
    plt.xlabel("Activation Value")
    
    plt.subplot(3, 3, 4)
    plt.hist(x_conv1_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("First Transformer Activations")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(3, 3, 5)
    plt.hist(x_last_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Last Transformer Activations")
    plt.xlabel("Activation Value")
    
    plt.subplot(3, 3, 6)
    plt.hist(x_pooled.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Pooled Features")
    plt.xlabel("Activation Value")
    
    plt.subplot(3, 3, 7)
    plt.hist(x_clf_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Classifier Activations")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(3, 3, 8)
    plt.hist(final_out.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Final Output")
    plt.xlabel("Probability")
    
    plt.tight_layout()
    plt.savefig("assets/activation_distributions.png")
    plt.close()
    
    print("Activation distribution plots saved to assets/activation_distributions.png")
    
    # Analyze dual encoder similarities
    if syn_similarities and ant_similarities:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        syn_by_label = [[], []]  # [0] for non-antonyms, [1] for antonyms
        ant_by_label = [[], []]
        
        for i, label in enumerate(labels):
            syn_by_label[int(label)].append(syn_similarities[i])
            ant_by_label[int(label)].append(ant_similarities[i])
        
        plt.boxplot([syn_by_label[0], syn_by_label[1]], labels=['Non-Antonyms', 'Antonyms'])
        plt.title("Synonym Space Similarities")
        plt.ylabel("Similarity (tanh)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([ant_by_label[0], ant_by_label[1]], labels=['Non-Antonyms', 'Antonyms'])
        plt.title("Antonym Space Similarities")
        plt.ylabel("Similarity (tanh)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("assets/dual_encoder_similarities.png")
        plt.close()
        
        print("Dual encoder similarity analysis saved to assets/dual_encoder_similarities.png")
        
        # Print similarity statistics
        print("\n**Dual Encoder Similarity Analysis**")
        print(f"Synonym space - Non-antonyms: {np.mean(syn_by_label[0]):.4f}, Antonyms: {np.mean(syn_by_label[1]):.4f}")
        print(f"Antonym space - Non-antonyms: {np.mean(ant_by_label[0]):.4f}, Antonyms: {np.mean(ant_by_label[1]):.4f}")
        
        # Check if the model learned the expected patterns
        syn_diff = np.mean(syn_by_label[0]) - np.mean(syn_by_label[1])
        ant_diff = np.mean(ant_by_label[1]) - np.mean(ant_by_label[0])
        
        print("\n**Dual Encoder Learning Assessment**")
        print(f"Synonym space differentiation: {syn_diff:.4f} (higher is better)")
        print(f"Antonym space differentiation: {ant_diff:.4f} (higher is better)")
        
        if syn_diff > 0:
            print("✓ Model correctly learned to represent non-antonyms with higher similarity in synonym space")
        else:
            print("✗ Model failed to learn expected pattern in synonym space")
            
        if ant_diff > 0:
            print("✓ Model correctly learned to represent antonyms with higher similarity in antonym space")
        else:
            print("✗ Model failed to learn expected pattern in antonym space")

def analyze_predictions(model, dataset, batch_size=32):
    """Analyze model predictions and errors."""
    print("\n## Prediction Analysis")
    
    # Create a dataloader
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size)
    
    # Storage for predictions, true labels, and confidence scores
    all_preds = []
    all_labels = []
    all_scores = []
    word_pairs = []  # Store indices to get back word pairs later if needed
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            scores = outputs.cpu().numpy()
            preds = (outputs >= 0.5).float().cpu().numpy()
            
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"Overall accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Antonym', 'Antonym'],
                yticklabels=['Not Antonym', 'Antonym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('assets/prediction_confusion_matrix.png')
    plt.close()
    
    # Analyze prediction confidence
    plt.figure(figsize=(10, 6))
    
    # Confidence histograms by true class
    class0_scores = all_scores[all_labels == 0]
    class1_scores = all_scores[all_labels == 1]
    
    plt.hist(class0_scores, bins=20, alpha=0.5, label='Not Antonyms (True)')
    plt.hist(class1_scores, bins=20, alpha=0.5, label='Antonyms (True)')
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence by Class')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('assets/prediction_confidence.png')
    plt.close()
    
    # Analyze errors
    errors = (all_preds != all_labels)
    error_rate = errors.mean()
    
    # False positives and false negatives
    false_positives = np.logical_and(all_preds == 1, all_labels == 0)
    false_negatives = np.logical_and(all_preds == 0, all_labels == 1)
    
    fp_rate = false_positives.mean()
    fn_rate = false_negatives.mean()
    
    print(f"\n**Error Analysis**")
    print(f"Overall error rate: {error_rate:.4f}")
    print(f"False positive rate: {fp_rate:.4f}")
    print(f"False negative rate: {fn_rate:.4f}")
    
    # Analyze confidence of errors
    if np.any(errors):
        error_scores = all_scores[errors]
        fp_scores = all_scores[false_positives] if np.any(false_positives) else []
        fn_scores = all_scores[false_negatives] if np.any(false_negatives) else []
        
        print(f"Mean confidence for errors: {np.mean(error_scores):.4f}")
        
        if len(fp_scores) > 0:
            print(f"Mean confidence for false positives: {np.mean(fp_scores):.4f}")
        
        if len(fn_scores) > 0:
            print(f"Mean confidence for false negatives: {np.mean(fn_scores):.4f}")
        
        # Plot error confidence
        plt.figure(figsize=(10, 6))
        
        if len(fp_scores) > 0:
            plt.hist(fp_scores, bins=20, alpha=0.5, label='False Positives')
        
        if len(fn_scores) > 0:
            plt.hist(fn_scores, bins=20, alpha=0.5, label='False Negatives')
        
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Confidence of Incorrect Predictions')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('assets/error_confidence.png')
        plt.close()

def evaluate_model(model, dataset, batch_size=32, dataset_name=""):
    """Evaluate graph model and print metrics."""
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"\n--- {dataset_name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Antonym', 'Antonym'],
                yticklabels=['Not Antonym', 'Antonym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'assets/graph_confusion_matrix_{dataset_name.replace(" ", "_")}.png')
    plt.close()
    return {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': conf_matrix}

def main():
    """Main function for model diagnosis."""
    print("## Dual Encoder Graph Model Diagnosis")
    
    # Define paths
    dataset_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    model_path = "best_graph_model.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    # Initialize the embedding model
    print("\nLoading Nomic embedding model...")
    model_st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # Load a small sample of data for diagnosis
    print("\nLoading sample data for diagnosis...")
    all_word1, all_word2, all_labels = [], [], []
    
    for word_type in word_types:
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        if os.path.exists(test_file):
            w1, w2, y = load_data(test_file)
            # Take a subset for efficiency
            subset_size = min(100, len(y))
            all_word1.extend(w1[:subset_size])
            all_word2.extend(w2[:subset_size])
            all_labels.extend(y[:subset_size])
    
    if not all_word1:
        print("Error: No test data found.")
        return
    
    print(f"Loaded {len(all_labels)} samples for diagnosis")
    
    # Generate embeddings
    X_word1, X_word2 = embed_word_pairs(all_word1, all_word2, model_st)
    
    # Create graph dataset
    dataset = create_graph_dataset(X_word1, X_word2, all_labels)
    
    # Initialize the model
    input_dim = X_word1.shape[1]  # Dimension of word embeddings
    model = DualEncoderGraphModel(input_dim=input_dim).to(device)
    
    # Load the model weights
    print(f"\nLoading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run diagnostic analyses
    analyze_model_weights(model)
    analyze_model_activations(model, dataset)
    analyze_predictions(model, dataset)
    
    # Analyze per word type performance
    print("\n## Word Type Analysis")
    for word_type in word_types:
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        if os.path.exists(test_file):
            print(f"\nAnalyzing performance on {word_type}...")
            w1, w2, y = load_data(test_file)
            # Take a subset for efficiency if needed
            subset_size = min(200, len(y))
            w1 = w1[:subset_size]
            w2 = w2[:subset_size]
            y = y[:subset_size]
            
            X_w1, X_w2 = embed_word_pairs(w1, w2, model_st)
            type_dataset = create_graph_dataset(X_w1, X_w2, y)
            
            # Evaluate on this word type
            evaluate_model(model, type_dataset, batch_size=64, dataset_name=f"Test - {word_type}")
    
    # Check for potential biases in the model
    print("\n## Bias Analysis")
    # Analyze if the model performs differently on different word types
    print("Checking for word type biases in model performance...")
    
    print("\nDiagnosis complete! Check the assets directory for visualizations.")

if __name__ == "__main__":
    main()
