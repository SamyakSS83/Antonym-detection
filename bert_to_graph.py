import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# -----------------------
# Device Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load Fine-Tuned BERT for Embeddings
# -----------------------
# Update this path to point to your fine-tuned BERT model checkpoint directory.
finetuned_bert_path = "good_models/bert/best_bert_model.pt"  # <-- CHANGE THIS PATH as needed

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load base model first, then load your saved weights
ft_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
ft_model.load_state_dict(torch.load(finetuned_bert_path, map_location=device))
# Extract the underlying BERT encoder for embeddings
finetuned_bert = ft_model.bert.to(device)
finetuned_bert.eval()

# -----------------------
# Data Loading Functions
# -----------------------

def load_data(file_path):
    """Load data from a file and return a DataFrame."""
    word1_list, word2_list, labels = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    return pd.DataFrame({'word1': word1_list, 'word2': word2_list, 'label': labels})

# -----------------------
# Graph Dataset Definition
# -----------------------

class WordPairGraphDataset(Dataset):
    """
    For each word pair, we create a graph with two nodes:
      - Each node is the BERT embedding (CLS token) of the word from your fine-tuned model.
      - A bidirectional edge connects the two nodes.
    The graph's label is the word pair's label.
    """
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word1 = self.data.iloc[idx]['word1']
        word2 = self.data.iloc[idx]['word2']
        label = self.data.iloc[idx]['label']

        # Use the fine-tuned BERT encoder to get embeddings for each word
        with torch.no_grad():
            emb1 = finetuned_bert(**tokenizer(word1, return_tensors='pt').to(device)).last_hidden_state[:, 0, :].squeeze(0).cpu()
            emb2 = finetuned_bert(**tokenizer(word2, return_tensors='pt').to(device)).last_hidden_state[:, 0, :].squeeze(0).cpu()

        # Node features for both words
        x = torch.stack([emb1, emb2])
        # Create a bidirectional edge between the two nodes
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        # Create a GraphData object from torch_geometric
        return GraphData(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))

# -----------------------
# Graph Transformer Model Definition
# -----------------------

class GraphTransformerClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphTransformerClassifier, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=2, dropout=0.2)
        self.conv2 = TransformerConv(2 * hidden_channels, hidden_channels, heads=1)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # data.x shape: [num_nodes, in_channels]
        # data.edge_index shape: [2, num_edges]
        # data.batch shape: [num_nodes] (provided by GeoDataLoader when batching graphs)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # Pooling to get graph-level embedding (mean over nodes per graph)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# -----------------------
# Load Dataset from /kaggle/input/dataset
# -----------------------

data_dir = "./dataset"
word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_sets = {}

for wt in word_types:
    train_file = os.path.join(data_dir, f"{wt}.train")
    val_file = os.path.join(data_dir, f"{wt}.val")
    test_file = os.path.join(data_dir, f"{wt}.test")
    
    if os.path.exists(train_file):
        train_df = pd.concat([train_df, load_data(train_file)], ignore_index=True)
    if os.path.exists(val_file):
        val_df = pd.concat([val_df, load_data(val_file)], ignore_index=True)
    if os.path.exists(test_file):
        test_sets[wt] = load_data(test_file)

# Combine training and validation sets for training
train_df = pd.concat([train_df, val_df], ignore_index=True)

# -----------------------
# Create Datasets and DataLoaders
# -----------------------

train_dataset = WordPairGraphDataset(train_df)
train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)

# Combined test dataset
combined_test_df = pd.concat(list(test_sets.values()), ignore_index=True)
combined_test_dataset = WordPairGraphDataset(combined_test_df)
combined_test_loader = GeoDataLoader(combined_test_dataset, batch_size=32)

# Test loaders per word type
test_loaders = {
    wt: GeoDataLoader(WordPairGraphDataset(df), batch_size=32)
    for wt, df in test_sets.items()
}

# -----------------------
# Model, Optimizer, and Loss Setup
# -----------------------

model = GraphTransformerClassifier(in_channels=768, hidden_channels=256, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def plot_confusion(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Antonym", "Antonym"], yticklabels=["Not Antonym", "Antonym"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------
# Training Loop
# -----------------------

num_epochs = 10
best_model_path = "assets/best_graph_transformer_model.pt"
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # Evaluate on combined test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in combined_test_loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Combined Test Accuracy: {acc:.4f}")
    
    # Save best model and plot confusion matrix if improved
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), best_model_path)
        print("Saved new best model.")
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion(cm, 'Combined Test Confusion Matrix', os.path.join('assets', 'combined_confusion_matrix.png'))

# -----------------------
# Evaluation on Each Word Type
# -----------------------

print("\nEvaluating best model on each word type:")
model.load_state_dict(torch.load(best_model_path))
model.eval()

for wt, loader in test_loaders.items():
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{wt} Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Not Antonym", "Antonym"]))
    
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion(cm, f"{wt} Confusion Matrix", os.path.join("assets", f"{wt}_confusion_matrix.png"))
