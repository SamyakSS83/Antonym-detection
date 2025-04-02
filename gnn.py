import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class GraphTransformer(nn.Module):
    """Graph Transformer model for antonym detection."""
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
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"\n--- {dataset_name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Antonym', 'Antonym'],
                yticklabels=['Not Antonym', 'Antonym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'graph_confusion_matrix_{dataset_name.replace(" ", "_")}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def train_model(model, train_dataset, val_dataset=None, epochs=10, batch_size=32, learning_rate=1e-3):
    """Train the graph model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({'loss': total_loss / train_batches})
        
        avg_train_loss = total_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation if provided
        if val_dataset:
            model.eval()
            total_val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in val_pbar:
                    batch = batch.to(device)
                    
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    val_pbar.set_postfix({'loss': total_val_loss / val_batches})
            
            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_graph_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # Plot training/validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_dataset:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('graph_training_loss.png')
    plt.close()
    
    return train_losses, val_losses

def main():
    # Define paths
    dataset_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    batch_size = 64
    epochs = 15
    
    # Initialize the embedding model
    print("Loading Nomic embedding model...")
    model_st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # Collect all training and validation data across all word types
    all_train_val_word1, all_train_val_word2, all_train_val_labels = [], [], []
    test_data_by_type = {}
    
    for word_type in word_types:
        # Load training data
        train_file = os.path.join(dataset_dir, f"{word_type}.train")
        val_file = os.path.join(dataset_dir, f"{word_type}.val")
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        
        w1_train, w2_train, y_train = load_data(train_file)
        w1_val, w2_val, y_val = load_data(val_file)
        w1_test, w2_test, y_test = load_data(test_file)
        
        # Add to combined training and validation data
        all_train_val_word1.extend(w1_train + w1_val)
        all_train_val_word2.extend(w2_train + w2_val)
        all_train_val_labels.extend(y_train + y_val)
        
        # Store test data separately for domain-wise evaluation
        test_data_by_type[word_type] = (w1_test, w2_test, y_test)
    
    print(f"Combined training and validation data: {len(all_train_val_labels)} samples")
    
    # Generate embeddings for training/validation
    X_train_val_word1, X_train_val_word2 = embed_word_pairs(all_train_val_word1, all_train_val_word2, model_st)
    
    # Create graph dataset for training/validation
    all_train_val_dataset = create_graph_dataset(X_train_val_word1, X_train_val_word2, all_train_val_labels)
    
    # Split into training and validation sets (90% train, 10% validation)