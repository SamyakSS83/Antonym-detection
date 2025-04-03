import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AntonymDataset(Dataset):
    """Custom dataset for antonym word pairs"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and split into heads
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.embed_dim)
        out = self.fc_out(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):  # Increased dropout from 0.1 to 0.3
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and normalization
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network with residual connection and normalization
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class AntonymTransformer(nn.Module):
    """Transformer model for antonym detection"""
    def __init__(self, input_dim, num_layers=1, num_heads=2, ff_dim=256, dropout=0.4):  # Reduced complexity & increased dropout
        super(AntonymTransformer, self).__init__()
        
        self.embed_dim = input_dim
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(input_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, ff_dim),
            nn.BatchNorm1d(ff_dim),  # Added BatchNorm
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, 2, embedding_dim]
        # Process each word embedding through transformer blocks
        word1 = x[:, 0, :]  # [batch_size, embedding_dim]
        word2 = x[:, 1, :]  # [batch_size, embedding_dim]
        
        # Add sequence dimension for transformer
        word1 = word1.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        word2 = word2.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            word1 = block(word1)
            word2 = block(word2)
        
        # Get the output embeddings
        word1_emb = word1.squeeze(1)  # [batch_size, embedding_dim]
        word2_emb = word2.squeeze(1)  # [batch_size, embedding_dim]
        
        # Concatenate embeddings for classification
        concat_emb = torch.cat((word1_emb, word2_emb), dim=1)
        
        # Classification
        output = self.classifier(concat_emb)
        
        return output.squeeze()

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
    
    # Stack embeddings of both words
    embeddings = np.stack([emb1, emb2], axis=1)
    
    print(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings

def evaluate_model(model, dataloader, dataset_name=""):
    """Evaluate model and print metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
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
    plt.savefig(f'assets/transformer_confusion_matrix_{dataset_name.replace(" ", "_")}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def train_model(model, train_dataloader, val_dataloader=None, epochs=10, learning_rate=5e-5):  # Reduced learning rate
    """Train the model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # Added LR scheduler
    
    best_val_loss = float('inf')
    patience = 15  # Reduced patience from 50 to 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        train_batches = 0
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({'loss': total_loss / train_batches})
        
        avg_train_loss = total_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation if provided
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for inputs, labels in val_pbar:
                    inputs = inputs.to(device)
                    labels = labels.float().to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    val_pbar.set_postfix({'loss': total_val_loss / val_batches})
            
            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_transformer_model.pt")
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
    if val_dataloader:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('assets/transformer_training_loss.png')
    plt.close()
    
    return train_losses, val_losses

def perform_kfold_cv(X, y, input_dim, k=5, batch_size=32, epochs=200, learning_rate=5e-5):
    """Perform k-fold cross-validation."""
    from sklearn.model_selection import KFold
    
    print(f"\n=== Performing {k}-fold Cross-Validation ===")
    
    # Initialize KFold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Lists to store metrics
    fold_val_losses = []
    fold_val_accuracies = []
    best_model_state = None
    best_val_loss = float('inf')
    
    # Iterate through folds
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        # Split data
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        # Create datasets and dataloaders
        train_dataset = AntonymDataset(X_train_fold, y_train_fold)
        val_dataset = AntonymDataset(X_val_fold, y_val_fold)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = AntonymTransformer(input_dim).to(device)
        
        # Train model
        train_losses, val_losses = train_model(
            model, train_dataloader, val_dataloader, 
            epochs=epochs, learning_rate=learning_rate
        )
        
        # Evaluate the trained model on validation fold
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0
        val_batches = 0
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                preds = (outputs >= 0.5).float().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                total_val_loss += loss.item()
                val_batches += 1
        
        fold_val_loss = total_val_loss / val_batches
        fold_accuracy = accuracy_score(all_labels, all_preds)
        
        # Store metrics
        fold_val_losses.append(fold_val_loss)
        fold_val_accuracies.append(fold_accuracy)
        
        print(f"Fold {fold+1} - Validation Loss: {fold_val_loss:.4f}, Accuracy: {fold_accuracy:.4f}")
        
        # Keep the best model
        if fold_val_loss < best_val_loss:
            best_val_loss = fold_val_loss
            best_model_state = model.state_dict().copy()
    
    # Print average results
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
    avg_val_accuracy = sum(fold_val_accuracies) / len(fold_val_accuracies)
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Average validation loss: {avg_val_loss:.4f}")
    print(f"Average validation accuracy: {avg_val_accuracy:.4f}")
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, "best_transformer_model_cv.pt")
        print("Best model saved to 'best_transformer_model_cv.pt'")
    
    return best_model_state, avg_val_loss, avg_val_accuracy

def main():
    # Define paths
    dataset_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    batch_size = 32  # Reduced batch size from 64 to 32
    epochs = 200
    k_folds = 5  # Number of folds for cross-validation
    
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
    X_train_val = embed_word_pairs(all_train_val_word1, all_train_val_word2, model_st)
    y_train_val = np.array(all_train_val_labels)
    
    # Perform k-fold cross-validation
    input_dim = X_train_val.shape[2]  # Embedding dimension
    best_model_state, _, _ = perform_kfold_cv(
        X_train_val, y_train_val, input_dim,
        k=k_folds, batch_size=batch_size, epochs=epochs
    )
    
    # Initialize model with the best weights from cross-validation
    model = AntonymTransformer(input_dim).to(device)
    model.load_state_dict(torch.load("best_transformer_model_cv.pt"))
    
    # Evaluate model on each domain's test set
    print("\n=== Evaluating Transformer Model on Test Sets by Domain ===")
    test_results = {}
    
    for word_type, (w1_test, w2_test, y_test) in test_data_by_type.items():
        print(f"\nEvaluating on {word_type} test set...")
        X_test = embed_word_pairs(w1_test, w2_test, model_st)
        
        test_dataset = AntonymDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        results = evaluate_model(model, test_dataloader, dataset_name=f"Transformer on {word_type}")
        test_results[word_type] = results
    
    # Calculate and print overall metrics
    all_accuracies = [results['accuracy'] for results in test_results.values()]
    avg_accuracy = np.mean(all_accuracies)
    
    print("\n=== Overall Results for Transformer Model ===")
    print(f"Average accuracy across all word types: {avg_accuracy:.4f}")
    
    for word_type, results in test_results.items():
        print(f"{word_type} accuracy: {results['accuracy']:.4f}")
    
    # Save results to CSV
    results_data = []
    for word_type, results in test_results.items():
        results_data.append({
            'Word Type': word_type,
            'Accuracy': results['accuracy']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('transformer_results.csv', index=False)
    print("Results saved to transformer_results.csv")

if __name__ == "__main__":
    main()