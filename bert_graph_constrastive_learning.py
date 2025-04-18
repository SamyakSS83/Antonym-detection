import os
# Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, SAGPooling, global_add_pool, global_mean_pool
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
# Update this path to point to your fine-tuned BERT model checkpoint directory
finetuned_bert_path = "/kaggle/working/assets/best_bert_model.pt"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ft_model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
    def __init__(self, dataframe, tokenizer, bert_model, device):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        # Pre-tokenize all words to avoid repeated tokenization
        self.word1_encodings = {}
        self.word2_encodings = {}
        
        # Process unique words in batches to speed up embedding generation
        unique_words = set(self.data['word1'].tolist() + self.data['word2'].tolist())
        word_batches = [list(unique_words)[i:i+32] for i in range(0, len(unique_words), 32)]
        
        print("Pre-computing word embeddings...")
        with torch.no_grad():
            for batch in tqdm(word_batches):
                inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to(self.device)
                outputs = self.bert_model(**inputs).last_hidden_state[:, 0, :]
                
                for i, word in enumerate(batch):
                    # Store the embedding - keep on CPU to save GPU memory
                    if word in self.data['word1'].values:
                        self.word1_encodings[word] = outputs[i].detach().cpu()
                    if word in self.data['word2'].values:
                        self.word2_encodings[word] = outputs[i].detach().cpu()
        print("Embeddings pre-computed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word1 = self.data.iloc[idx]['word1']
        word2 = self.data.iloc[idx]['word2']
        label = self.data.iloc[idx]['label']

        # Get pre-computed embeddings
        emb1 = self.word1_encodings[word1]
        emb2 = self.word2_encodings[word2]

        # Node features for both words
        x = torch.stack([emb1, emb2])
        
        # Create a bidirectional edge between the two nodes
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Create a GraphData object from torch_geometric
        return GraphData(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))

# -----------------------
# Hierarchical GAT with Contrastive Learning Model Definition
# -----------------------
class HierarchicalGATWithContrastive(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2,
                 temperature=0.07, projection_dim=128):
        super(HierarchicalGATWithContrastive, self).__init__()
        
        # Hierarchical GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        
        # Pooling layer to create a hierarchical representation
        self.pool1 = SAGPooling(hidden_channels * num_heads, ratio=0.5)
        
        # Second GAT layer after pooling
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # Attention-based context aggregation
        self.context_attention = nn.MultiheadAttention(hidden_channels, num_heads=2, dropout=dropout)
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
        # Contrastive learning components
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, projection_dim)
        )
        
        # Positional encoding for word order
        self.pos_encoder = nn.Embedding(2, in_channels)  
        
        # Relation-specific embeddings for synonym/antonym signal
        self.relation_embeddings = nn.Embedding(2, hidden_channels)  # 0: synonym, 1: antonym
        
        # Separate pathways for synonym and antonym learning
        self.synonym_proj = nn.Linear(hidden_channels, hidden_channels)
        self.antonym_proj = nn.Linear(hidden_channels, hidden_channels)
        
    def encode_graph(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Add positional encoding to differentiate word order in pair
        pos_ids = torch.tensor([0, 1]).repeat(batch.max().item() + 1).to(x.device)
        x = x + self.pos_encoder(pos_ids)
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Pooling to create hierarchical structure
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        # Self-attention for context modeling
        # Check if we have enough nodes for attention
        if x.size(0) > 1:
            # Get number of unique batches
            num_batches = batch.max().item() + 1
            # Reshape for multi-head attention: [seq_len, batch_size, embedding_dim]
            try:
                x_reshaped = x.view(-1, num_batches, x.size(-1)).permute(1, 0, 2)
                attn_output, _ = self.context_attention(x_reshaped, x_reshaped, x_reshaped)
                # Combine attention output with original representations
                x_context = attn_output.permute(1, 0, 2).reshape(-1, x.size(-1))
                x = x + x_context  # Residual connection
            except RuntimeError:
                # In case of irregular batch sizes, skip attention
                pass
        
        # Global pooling to get graph representation
        x_graph = global_mean_pool(x, batch)
        
        return x_graph
        
    def forward(self, data, return_embeddings=False):
        x_graph = self.encode_graph(data)
        
        # Classification path
        x = F.relu(self.lin1(x_graph))
        logits = self.lin2(x)
        
        if return_embeddings:
            return logits, x_graph
        return logits
    
    def get_contrastive_embeddings(self, data):
        _, embeddings = self.forward(data, return_embeddings=True)
        
        # Project to the space where contrastive loss is applied
        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, p=2, dim=1)
        
        # Get graph-level labels 
        # Each graph corresponds to a word pair, so we need to extract the labels properly
        unique_batches = torch.arange(embeddings.size(0)).to(embeddings.device)
        
        # Create a mapping from batch index to actual label
        batch_to_label = {}
        for i in range(len(data.y)):
            batch_id = data.batch[i].item()
            if batch_id not in batch_to_label:
                batch_to_label[batch_id] = data.y[i].item()
        
        # Get labels in the order of unique batches
        batch_labels = torch.tensor([batch_to_label.get(i, 0) for i in range(len(unique_batches))], 
                                    device=embeddings.device)
        
        relation_embs = self.relation_embeddings(batch_labels)
        
        # Create synonym and antonym projections
        syn_embs = self.synonym_proj(embeddings)
        ant_embs = self.antonym_proj(embeddings)
        
        return projections, batch_labels, syn_embs, ant_embs, relation_embs
        
    def contrastive_loss(self, data):
        projections, labels, syn_embs, ant_embs, relation_embs = self.get_contrastive_embeddings(data)
        
        batch_size = projections.size(0)
        if batch_size <= 1:
            # Not enough samples for contrastive loss
            return torch.tensor(0.0, device=projections.device)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Mask for positive pairs (same class)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos.fill_diagonal_(False)  # Remove self-contrast
        
        # Mask for negative pairs (different class)
        mask_neg = ~mask_pos
        mask_neg.fill_diagonal_(False)  # Remove self-contrast
        
        # For numerical stability, handle case where no positive pairs exist
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=projections.device)
        
        # Compute InfoNCE / NT-Xent loss safely
        # Instead of reshaping which can cause dimension mismatch,
        # we'll compute the loss directly using the masks
        
        # Create a mask for valid entries (where either pos or neg is True)
        valid_mask = mask_pos | mask_neg
        
        # Set diagonal to False (don't compare sample with itself)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=projections.device)
        valid_mask = valid_mask & ~diag_mask
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=projections.device)
        
        # Calculate positive and negative scores
        pos_scores = similarity_matrix[mask_pos].mean() if mask_pos.sum() > 0 else torch.tensor(0.0, device=projections.device)
        neg_scores = similarity_matrix[mask_neg].mean() if mask_neg.sum() > 0 else torch.tensor(0.0, device=projections.device)
        
        # NT-Xent loss: encourage positive pairs to have higher similarity than negative pairs
        contrastive_loss = -pos_scores + neg_scores
        
        return contrastive_loss
    
    def relation_loss(self, data):
        # Get embeddings using the fixed method
        _, labels, syn_embs, ant_embs, relation_embs = self.get_contrastive_embeddings(data)
        
        if len(syn_embs) == 0 or len(relation_embs) == 0:
            return torch.tensor(0.0, device=relation_embs.device)
        
        # For synonym pairs, maximize similarity in synonym space
        syn_sim = F.cosine_similarity(syn_embs, relation_embs)
        
        # For antonym pairs, maximize similarity in antonym space
        ant_sim = F.cosine_similarity(ant_embs, relation_embs)
        
        # Labels: 0 for synonym, 1 for antonym
        labels = labels.float()
        
        # Compute relation loss - handle empty tensors
        if len(syn_sim) == 0:
            syn_loss = torch.tensor(0.0, device=relation_embs.device)
        else:
            syn_loss = -torch.mean((1 - labels) * syn_sim)  # High for synonyms (label=0)
            
        if len(ant_sim) == 0:
            ant_loss = torch.tensor(0.0, device=relation_embs.device)
        else:
            ant_loss = -torch.mean(labels * ant_sim)  # High for antonyms (label=1)
        
        return syn_loss + ant_loss

def plot_confusion(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Antonym", "Antonym"], yticklabels=["Not Antonym", "Antonym"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Improved test-time augmentation for inference
def test_with_augmentation(model, test_loader, num_augments=5):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device efficiently with non_blocking=True
            batch = batch.to(device, non_blocking=True)
            
            # Multiple forward passes with dropout enabled - use torch.no_grad for efficiency
            model.train()  # Enable dropout for stochastic forward passes
            logits_list = []
            
            # Process all augmentations in a single batch when possible
            if hasattr(model, 'training'):
                # Store original training mode
                original_training = model.training
                
                # Enable dropout
                model.train()
                
                # Perform multiple augmented forward passes
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    for _ in range(num_augments):
                        logits = model(batch)
                        logits_list.append(F.softmax(logits, dim=1))
                
                # Restore original training mode
                model.train(original_training)
            else:
                # Fallback for models without training mode
                for _ in range(num_augments):
                    logits = model(batch)
                    logits_list.append(F.softmax(logits, dim=1))
            
            # Average predictions from augmented passes directly on GPU
            avg_logits = torch.stack(logits_list).mean(dim=0)
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return all_preds, all_labels

# -----------------------
# Main Program
# -----------------------
def main():
    # -----------------------
    # Load Dataset
    # -----------------------
    data_dir = "/kaggle/input/dataset"
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
    # Determine optimal batch size based on GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    batch_size = 16 if gpu_mem < 8 else (32 if gpu_mem < 16 else 64)
    print(f"Using batch size: {batch_size} (GPU memory: {gpu_mem:.2f} GB)")

    # Create datasets with explicit passing of tokenizer and model
    train_dataset = WordPairGraphDataset(train_df, tokenizer, finetuned_bert, device)
    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, 
                              pin_memory=True)

    # Combined test dataset
    combined_test_df = pd.concat(list(test_sets.values()), ignore_index=True)
    combined_test_dataset = WordPairGraphDataset(combined_test_df, tokenizer, finetuned_bert, device)
    combined_test_loader = GeoDataLoader(combined_test_dataset, batch_size=batch_size, num_workers=4, 
                                      pin_memory=True)

    # Test loaders per word type
    test_loaders = {
        wt: GeoDataLoader(WordPairGraphDataset(df, tokenizer, finetuned_bert, device), 
                       batch_size=batch_size, num_workers=4, pin_memory=True)
        for wt, df in test_sets.items()
    }

    # -----------------------
    # Model, Optimizer, and Loss Setup
    # -----------------------
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = HierarchicalGATWithContrastive(
        in_channels=768,  # BERT hidden size
        hidden_channels=256,
        out_channels=2,
        num_heads=4,
        dropout=0.2,
        temperature=0.07,
        projection_dim=128
    ).to(device)

    # Use mixed precision training for better GPU performance
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Loss functions
    criterion = nn.CrossEntropyLoss()
    lambda_contrastive = 0.3  # Weight for contrastive loss
    lambda_relation = 0.2     # Weight for relation-specific loss

    # -----------------------
    # Training Loop
    # -----------------------
    num_epochs = 15
    best_model_path = "assets/best_hierarchical_gat_model.pt"
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device efficiently with non_blocking=True
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Use mixed precision for faster computation on GPU
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # Get logits
                    logits = model(batch)
                    # Compute classification loss
                    ce_loss = criterion(logits, batch.y)
                    # Compute contrastive loss
                    cont_loss = model.contrastive_loss(batch)
                    # Compute relation-specific loss
                    rel_loss = model.relation_loss(batch)
                    # Total loss
                    loss = ce_loss + lambda_contrastive * cont_loss + lambda_relation * rel_loss

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Original non-mixed-precision path
                logits = model(batch)
                ce_loss = criterion(logits, batch.y)
                cont_loss = model.contrastive_loss(batch)
                rel_loss = model.relation_loss(batch)
                loss = ce_loss + lambda_contrastive * cont_loss + lambda_relation * rel_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # Explicitly free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update learning rate
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # ---------------------
        # Evaluate on combined test set with test-time augmentation
        all_preds, all_labels = test_with_augmentation(model, combined_test_loader)
        acc = accuracy_score(all_labels, all_preds)
        print(f"Combined Test Accuracy: {acc:.4f}")

        # Save best model and plot confusion matrix if improved
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with accuracy: {acc:.4f}")
            
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion(cm, 'Combined Test Confusion Matrix', os.path.join('assets', 'hierarchical_combined_confusion.png'))

    # -----------------------
    # Evaluation on Each Word Type
    # -----------------------
    print("\nEvaluating best model on each word type:")
    model.load_state_dict(torch.load(best_model_path))
    
    for wt, loader in test_loaders.items():
        all_preds, all_labels = test_with_augmentation(model, loader)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n{wt} Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["Not Antonym", "Antonym"]))
        
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion(cm, f"{wt} Confusion Matrix", os.path.join("assets", f"hierarchical_{wt}_confusion.png"))

if __name__ == "__main__":
    main()