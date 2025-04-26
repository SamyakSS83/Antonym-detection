import os
# Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, GINConv, global_mean_pool
from torch_geometric.data import Data as GraphData
import numpy as np
from functools import partial

import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GeoDataLoader
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


class MultiScaleGraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2, 
                 heads=4, temp=0.07, project_dim=128):
        super(MultiScaleGraphTransformer, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.temp = temp
        self.project_dim = project_dim
        
        # Word embedding projection
        self.word_proj = nn.Linear(in_channels, hidden_channels)
        
        # Calculate total features after concatenation
        self.total_features = hidden_channels * heads + hidden_channels + hidden_channels
        
        # Multiple scales of graph convolutions
        self.conv_layers = nn.ModuleList()
        # Add dimension reduction projections for next layer input
        self.dim_reduce = nn.ModuleList()
        
        for i in range(num_layers):
            # For first layer, input is hidden_channels
            # For subsequent layers, input is also hidden_channels (after projection)
            layer = nn.ModuleDict({
                'transformer': TransformerConv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    dropout=dropout,
                    beta=True,
                    concat=True
                ),
                'gcn': GCNConv(
                    hidden_channels, 
                    hidden_channels
                ),
                'gin': GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels)
                    )
                )
            })
            self.conv_layers.append(layer)
            
            # Add projection layer to reduce dimensions back to hidden_channels
            self.dim_reduce.append(nn.Linear(self.total_features, hidden_channels))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.total_features) for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attn = nn.MultiheadAttention(self.total_features, num_heads=heads, dropout=dropout)
        
        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Linear(self.total_features, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
        # Auxiliary classification heads for intermediate supervision
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(self.total_features, out_channels) for _ in range(num_layers-1)
        ])
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Add domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 3)  # 3 classes: noun, verb, adjective
        )
        
        # Word type embedding
        self.word_type_embedding = nn.Embedding(3, hidden_channels // 2)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, project_dim)
        )

    def forward(self, data, word_type_ids=None, alpha=1.0):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial projection
        x = self.word_proj(x)
        
        aux_outputs = []
        
        # Multi-scale message passing
        for i in range(self.num_layers):
            # Apply different types of convolutions
            x1 = self.conv_layers[i]['transformer'](x, edge_index)
            x2 = self.conv_layers[i]['gcn'](x, edge_index)
            x3 = self.conv_layers[i]['gin'](x, edge_index)
            
            # Concatenate multi-scale features
            x_cat = torch.cat([x1, x2, x3], dim=-1)
            
            # Apply layer normalization
            x_norm = self.layer_norms[i](x_cat)
            
            # Store for auxiliary output if needed
            if i < self.num_layers - 1:
                graph_x = global_mean_pool(x_norm, batch)
                aux_out = self.aux_classifiers[i](graph_x)
                aux_outputs.append(aux_out)
            
            # Project back to hidden_channels for next layer input
            x = self.dim_reduce[i](x_norm)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Use the final concatenated features for global attention
        x_final = x_norm  # This has dimension total_features
        
        # Global attention pooling
        num_nodes = torch.bincount(batch)
        max_nodes = max(num_nodes).item()
        
        # Create a padded tensor for attention
        padded_x = torch.zeros(batch.max().item() + 1, max_nodes, x_final.size(-1), device=x_final.device)
        
        # Fill the padded tensor
        node_offset = 0
        for i, count in enumerate(num_nodes):
            padded_x[i, :count] = x_final[node_offset:node_offset+count]
            node_offset += count
        
        # Apply global attention
        padded_x_t = padded_x.transpose(0, 1)  # [max_nodes, batch_size, feat_dim]
        attn_output, _ = self.global_attn(padded_x_t, padded_x_t, padded_x_t)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, max_nodes, feat_dim]
        
        # Create attention masks based on actual node counts
        attn_mask = torch.zeros(batch.max().item() + 1, max_nodes, device=x_final.device, dtype=torch.bool)
        for i, count in enumerate(num_nodes):
            attn_mask[i, count:] = True
        
        # Apply mask and pool
        masked_output = attn_output.masked_fill(attn_mask.unsqueeze(-1), 0)
        pooled_x = masked_output.sum(dim=1) / num_nodes.unsqueeze(-1).float()
        
        # Store the pooled attention output for contrastive learning
        self.global_attn_output = pooled_x
        
        # Feature extraction
        features = self.feat_extract(pooled_x)
        
        # Main classifier
        logits = self.classifier(features)
        
        outputs = {"main_logits": logits}
        if aux_outputs:
            outputs["aux_logits"] = aux_outputs
        
        # Domain classifier with gradient reversal
        if word_type_ids is not None:
            # Apply gradient reversal layer (done in training loop)
            domain_logits = self.domain_classifier(features.detach())
            outputs["domain_logits"] = domain_logits
            
            # Get word type embeddings to augment features
            word_type_emb = self.word_type_embedding(word_type_ids)
            augmented_features = torch.cat([features, word_type_emb], dim=-1)
            outputs["augmented_features"] = augmented_features
        
        return outputs
    
    def contrastive_loss(self, data):
        """Calculate contrastive loss between graph representations"""
        # Forward pass to get features
        outputs = self(data)
        
        # Get features from the global attention output
        features = self.global_attn_output if hasattr(self, 'global_attn_output') else self.feat_extract(outputs["main_logits"])
        
        # Project features to contrastive space
        projections = self.projection_head(self.feat_extract(features))
        projections = F.normalize(projections, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(projections, projections.T) / self.temp
        
        # Create labels: positives are from same class
        labels = data.y.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-contrast
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
        # Compute loss
        pos_mask = mask > 0
        neg_mask = mask == 0
        
        # Skip if no positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=similarity.device)
            
        pos_similarity = similarity[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=similarity.device)
        neg_similarity = similarity[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=similarity.device)
        
        loss = -pos_similarity + neg_similarity
        return loss
    
    def relation_loss(self, data):
        """Calculate relation-specific loss for antonym vs non-antonym distinctions"""
        # Get embeddings from the model
        outputs = self(data)
        features = self.feat_extract(self.global_attn_output) if hasattr(self, 'global_attn_output') else outputs["main_logits"]
        
        # Separate antonym and non-antonym pairs
        labels = data.y
        antonym_mask = (labels == 1)
        non_antonym_mask = (labels == 0)
        
        # Skip if no pairs of either type
        if not antonym_mask.any() or not non_antonym_mask.any():
            return torch.tensor(0.0, device=features.device)
        
        # Get embeddings for each type
        antonym_embs = features[antonym_mask]
        non_antonym_embs = features[non_antonym_mask]
        
        # Calculate centroids
        antonym_centroid = antonym_embs.mean(dim=0, keepdim=True)
        non_antonym_centroid = non_antonym_embs.mean(dim=0, keepdim=True)
        
        # Calculate within-class distance and between-class separation
        within_antonym = F.pairwise_distance(antonym_embs, antonym_centroid.repeat(antonym_embs.size(0), 1)).mean()
        within_non_antonym = F.pairwise_distance(non_antonym_embs, non_antonym_centroid.repeat(non_antonym_embs.size(0), 1)).mean()
        
        between_centroids = F.pairwise_distance(antonym_centroid, non_antonym_centroid)
        
        # Relation-specific loss: minimize within-class distance, maximize between-class separation
        rel_loss = (within_antonym + within_non_antonym) / (between_centroids + 1e-5)
        
        return rel_loss

# Loss function with multiple components
class MultiComponentLoss(nn.Module):
    def __init__(self, aux_weight=0.3, adv_weight=0.1):
        super(MultiComponentLoss, self).__init__()
        self.aux_weight = aux_weight
        self.adv_weight = adv_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, domain_targets=None):
        # Main classification loss
        main_loss = self.ce_loss(outputs["main_logits"], targets)
        
        total_loss = main_loss
        
        # Auxiliary losses for deep supervision
        if "aux_logits" in outputs:
            aux_loss = 0
            for aux_logit in outputs["aux_logits"]:
                aux_loss += self.ce_loss(aux_logit, targets)
            total_loss += self.aux_weight * aux_loss / len(outputs["aux_logits"])
        
        # Domain adversarial loss
        if "domain_logits" in outputs and domain_targets is not None:
            domain_loss = self.ce_loss(outputs["domain_logits"], domain_targets)
            # Gradient reversal is implemented by negating the loss
            total_loss += self.adv_weight * (-domain_loss)
        
        return total_loss

# Training function with adversarial examples
def train_multiscale_model(model, train_loader, optimizer, criterion, word_type_mapping, adv_prob=0.5, epsilon=0.01):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Map words to their types (noun, verb, adj) using the mapping
        # This is for domain adaptation
        word_types = []
        for idx in range(batch.num_graphs):
            # Extract the graph and get the first word
            start_idx = batch.ptr[idx].item()
            end_idx = batch.ptr[idx+1].item() if idx < batch.num_graphs - 1 else batch.x.size(0)
            if end_idx > start_idx:
                # Get first word from the graph
                # In practice, you'd have a mapping from words to their POS tags
                word_type = word_type_mapping.get(idx % 3, 0)  # Simplified mapping for example
                word_types.append(word_type)
        
        word_type_ids = torch.tensor(word_types, device=batch.x.device)
        
        # Generate adversarial examples with some probability
        if torch.rand(1).item() < adv_prob:
            batch_adv = model.get_adversarial_examples(batch, epsilon=epsilon)
            
            # Forward pass with adversarial examples
            outputs_adv = model(batch_adv, word_type_ids)
            
            # Standard forward pass
            outputs = model(batch, word_type_ids)
            
            # Combine losses
            loss = criterion(outputs, batch.y, word_type_ids) + criterion(outputs_adv, batch.y, word_type_ids)
            loss = loss / 2  # Average the losses
        else:
            # Standard forward pass only
            outputs = model(batch, word_type_ids)
            loss = criterion(outputs, batch.y, word_type_ids)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Inference with model ensemble
def ensemble_inference(models, test_loader):
    for model in models:
        model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Get predictions from all models
            ensemble_logits = []
            for model in models:
                outputs = model(batch)
                logits = outputs["main_logits"]
                probs = F.softmax(logits, dim=1)
                ensemble_logits.append(probs)
            
            # Average predictions
            avg_logits = torch.stack(ensemble_logits).mean(dim=0)
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return all_preds, all_labels

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
            
            # Multiple forward passes with dropout enabled
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
                        outputs = model(batch)
                        logits = outputs["main_logits"]  # Extract main logits from dict
                        logits_list.append(F.softmax(logits, dim=1))
                
                # Restore original training mode
                model.train(original_training)
            else:
                # Fallback for models without training mode
                for _ in range(num_augments):
                    outputs = model(batch)
                    logits = outputs["main_logits"]  # Extract main logits from dict
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

    model = MultiScaleGraphTransformer(
        in_channels=768,      # BERT embedding size
        hidden_channels=256,  # Hidden size for transformer
        out_channels=2,       # Number of classes (antonym, not antonym)
        num_layers=3,         # Number of graph convolution layers
        dropout=0.2,          # Dropout rate
        heads=4,              # Number of attention heads
        temp=0.5,             # Temperature for contrastive loss
        project_dim=128       # Projection dimension for contrastive loss
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
                    # Get model outputs
                    outputs = model(batch)  # Changed from logits = model(batch)
                    # Extract main logits
                    logits = outputs["main_logits"]
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
                outputs = model(batch)  # Changed from logits = model(batch)
                # Extract main logits
                logits = outputs["main_logits"]
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