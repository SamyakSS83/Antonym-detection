import numpy as np
import os
import torch
import torch.nn as nn
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
class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, heads=4):
        super(GraphTransformer, self).__init__()
        
        # Graph transformer layers
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, dropout=0.1)
        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
            for _ in range(num_layers - 1)
        ])
        
        # Output classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First graph transformer layer
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Additional graph transformer layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        
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

def analyze_model_weights(model):
    """Analyze the model weights and biases."""
    print("\n## Model Weight Analysis")
    
    # Analyze transformer layers
    print("\n**Transformer Layer Analysis**")
    
    # First layer
    w1 = model.conv1.lin_key.weight
    print(f"First layer key weights - Shape: {w1.shape}, Mean: {w1.mean().item():.4f}, Std: {w1.std().item():.4f}")
    
    # Last transformer layer
    last_w = model.convs[-1].lin_key.weight
    print(f"Last layer key weights - Shape: {last_w.shape}, Mean: {last_w.mean().item():.4f}, Std: {last_w.std().item():.4f}")
    
    # Classifier weights
    clf_w = model.classifier[0].weight
    print(f"Classifier weights - Shape: {clf_w.shape}, Mean: {clf_w.mean().item():.4f}, Std: {clf_w.std().item():.4f}")
    
    # Plot weight distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(w1.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("First Layer Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 2)
    plt.hist(last_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Last Layer Weights")
    plt.xlabel("Weight Value")
    
    plt.subplot(1, 3, 3)
    plt.hist(clf_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Classifier Weights")
    plt.xlabel("Weight Value")
    
    plt.tight_layout()
    plt.savefig("assets/weight_distributions.png")
    plt.close()
    
    print("Weight distribution plots saved to assets/weight_distributions.png")

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
        # First layer activation
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        # First transformer layer
        first_layer_out = model.conv1(x, edge_index)
        first_layer_act = model.activation(first_layer_out)
        
        # Last transformer layer
        last_layer_in = first_layer_act
        for i, conv in enumerate(model.convs[:-1]):
            last_layer_in = model.activation(conv(last_layer_in, edge_index))
        
        last_layer_out = model.convs[-1](last_layer_in, edge_index)
        last_layer_act = model.activation(last_layer_out)
        
        # Global pooling
        pooled = global_mean_pool(last_layer_act, batch_idx)
        
        # Classifier first layer
        clf_out = model.classifier[0](pooled)
        
        # Final output
        final_out = model(batch)
    
    # Analyze activations
    print("\n**Activation Statistics**")
    print(f"First layer output - Mean: {first_layer_out.mean().item():.4f}, Std: {first_layer_out.std().item():.4f}")
    print(f"First layer activation - Mean: {first_layer_act.mean().item():.4f}, Std: {first_layer_act.std().item():.4f}")
    print(f"Last layer activation - Mean: {last_layer_act.mean().item():.4f}, Std: {last_layer_act.std().item():.4f}")
    print(f"Pooled features - Mean: {pooled.mean().item():.4f}, Std: {pooled.std().item():.4f}")
    print(f"Classifier output - Mean: {clf_out.mean().item():.4f}, Std: {clf_out.std().item():.4f}")
    print(f"Final output - Mean: {final_out.mean().item():.4f}, Std: {final_out.std().item():.4f}")
    
    # Plot activation distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(first_layer_out.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("First Layer Output")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 3, 2)
    plt.hist(first_layer_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("First Layer Activation")
    plt.xlabel("Activation Value")
    
    plt.subplot(2, 3, 3)
    plt.hist(last_layer_act.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Last Layer Activation")
    plt.xlabel("Activation Value")
    
    plt.subplot(2, 3, 4)
    plt.hist(pooled.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Pooled Features")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 3, 5)
    plt.hist(clf_out.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Classifier Output")
    plt.xlabel("Activation Value")
    
    plt.subplot(2, 3, 6)
    plt.hist(final_out.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title("Final Output")
    plt.xlabel("Activation Value")
    
    plt.tight_layout()
    plt.savefig("assets/activation_distributions.png")
    plt.close()
    
    print("Activation distribution plots saved to assets/activation_distributions.png")

def analyze_predictions(model, dataset):
    """Analyze model predictions on the dataset."""
    print("\n## Prediction Analysis")
    
    # Create dataloader
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
    
    # Get predictions
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    print(f"\n**Classification Metrics**")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Antonym', 'Antonym'],
                yticklabels=['Not Antonym', 'Antonym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png")
    plt.close()
    
    print("Confusion matrix saved to assets/confusion_matrix.png")
    
    # Plot probability distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_probs, bins=50, alpha=0.7)
    plt.title("Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    correct = all_preds == all_labels
    plt.scatter(range(len(all_probs)), all_probs, c=correct, alpha=0.5, cmap='coolwarm')
    plt.colorbar(label='Correct Prediction')
    plt.title("Prediction Probabilities")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    
    plt.tight_layout()
    plt.savefig("assets/probability_analysis.png")
    plt.close()
    
    print("Probability analysis saved to assets/probability_analysis.png")
    
    # Analyze error cases
    error_indices = np.where(all_preds != all_labels)[0]
    print(f"\n**Error Analysis**")
    print(f"Number of errors: {len(error_indices)} out of {len(all_labels)} samples ({len(error_indices)/len(all_labels)*100:.2f}%)")
    
    # Analyze false positives and false negatives
    fp_indices = np.where((all_preds == 1) & (all_labels == 0))[0]
    fn_indices = np.where((all_preds == 0) & (all_labels == 1))[0]
    
    print(f"False positives: {len(fp_indices)} ({len(fp_indices)/len(all_labels)*100:.2f}%)")
    print(f"False negatives: {len(fn_indices)} ({len(fn_indices)/len(all_labels)*100:.2f}%)")
    
    # Plot error probability distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if len(fp_indices) > 0:
        plt.hist(all_probs[fp_indices], bins=20, alpha=0.7, color='red', label='False Positives')
    if len(fn_indices) > 0:
        plt.hist(all_probs[fn_indices], bins=20, alpha=0.7, color='blue', label='False Negatives')
    plt.title("Error Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(all_probs)), all_probs, c=all_labels, alpha=0.5, cmap='coolwarm')
    plt.colorbar(label='True Label')
    plt.title("Probabilities by True Label")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    
    plt.tight_layout()
    plt.savefig("assets/error_analysis.png")
    plt.close()
    
    print("Error analysis saved to assets/error_analysis.png")

def main():
    """Main function for model diagnosis."""
    print("## Graph Transformer Model Diagnosis")
    
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
    model = GraphTransformer(input_dim=input_dim).to(device)
    
    # Load the model weights
    print(f"\nLoading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run diagnostic analyses
    analyze_model_weights(model)
    analyze_model_activations(model, dataset)
    analyze_predictions(model, dataset)
    
    print("\nDiagnosis complete! Check the assets directory for visualizations.")

if __name__ == "__main__":
    main()
