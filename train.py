import nltk
print(1)
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
print(2)
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Optional: For a graph transformer, PyTorch Geometric might be needed:
# import torch_geometric
# from torch_geometric.nn import TransformerConv

# -------------------------------------------
# Data Gathering
# -------------------------------------------
def gather_antonym_pairs():
    """
    Gathers *all* antonym pairs from WordNet instead of limiting to sample_size.
    """
    antonym_pairs = []
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    antonym_pairs.append((lemma.name(), ant.name(), 1))

    # Generate the same number of negative samples
    all_words = list(set([pair[0] for pair in antonym_pairs] + [pair[1] for pair in antonym_pairs]))
    neg_samples = []
    while len(neg_samples) < len(antonym_pairs):
        w1, w2 = random.sample(all_words, 2)
        if w1 != w2 and (w1, w2, 1) not in antonym_pairs and (w2, w1, 1) not in antonym_pairs:
            neg_samples.append((w1, w2, 0))

    dataset = antonym_pairs + neg_samples
    random.shuffle(dataset)
    return dataset

# -------------------------------------------
# Model Architectures
# -------------------------------------------
# class SimpleAttentionNN(nn.Module):
#     def __init__(self, embed_dim=768):
#         super(SimpleAttentionNN, self).__init__()
#         self.attention_query = nn.Linear(embed_dim, embed_dim)
#         self.attention_key = nn.Linear(embed_dim, embed_dim)
#         self.attention_value = nn.Linear(embed_dim, embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: (batch_size, 2, embed_dim)
#         q = self.attention_query(x[:,0,:])  # (batch_size, embed_dim)
#         k = self.attention_key(x[:,1,:])    # (batch_size, embed_dim)
#         v = self.attention_value(x[:,1,:])  # (batch_size, embed_dim)

#         # Single-headed attention
#         attn_scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(2)  # (batch_size, 1)
#         attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, 1)
#         context = v * attn_weights  # (batch_size, embed_dim)

#         x_out = x[:,0,:] + context  # simple combination
#         return self.fc(x_out)

class SimpleAttentionNN(nn.Module):
    def __init__(self, embed_dim=768):
        super(SimpleAttentionNN, self).__init__()
        # Multi-head attention
        self.num_heads = 4
        head_dim = embed_dim // self.num_heads
        
        self.attention_query = nn.Linear(embed_dim, embed_dim)
        self.attention_key = nn.Linear(embed_dim, embed_dim)
        self.attention_value = nn.Linear(embed_dim, embed_dim)
        
        # Deeper classification layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, 2, embed_dim)
        batch_size = x.size(0)
        
        # Multi-head attention
        q = self.attention_query(x[:,0,:]).view(batch_size, self.num_heads, -1)
        k = self.attention_key(x[:,1,:]).view(batch_size, self.num_heads, -1)
        v = self.attention_value(x[:,1,:]).view(batch_size, self.num_heads, -1)

        # Scaled dot-product attention
        attn_scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, v).view(batch_size, -1)
        
        # Residual connection
        x_out = x[:,0,:] + context
        return self.fc(x_out)


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, 2, embed_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, 2, hidden_dim)
        last_timestep = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        return self.fc(last_timestep)

# Possible sketch of a GraphTransformer (if using PyTorch Geometric)
# class GraphTransformerModel(nn.Module):
#     def __init__(self, dataset_feature_dim):
#         super().__init__()
#         self.conv1 = TransformerConv(dataset_feature_dim, 64, heads=4)
#         self.conv2 = TransformerConv(64*4, 1, heads=1)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = nn.ReLU()(x)
#         x = self.conv2(x, edge_index)
#         return torch.sigmoid(x)

# -------------------------------------------
# Training Helpers
# -------------------------------------------
# def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=5, lr=1e-3):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     X_train_t = torch.tensor(X_train, dtype=torch.float32)
#     y_train_t = torch.tensor(y_train, dtype=torch.float32)
#     X_val_t = torch.tensor(X_val, dtype=torch.float32)
#     y_val_t = torch.tensor(y_val, dtype=torch.float32)

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train_t)
#         loss = criterion(outputs.squeeze(), y_train_t)
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_out = model(X_val_t).squeeze()
#             val_loss = criterion(val_out, y_val_t)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

class ImprovedAttentionNN(nn.Module):
    def __init__(self, embed_dim=768):
        super(ImprovedAttentionNN, self).__init__()
        self.num_heads = 8
        head_dim = embed_dim // self.num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, self.num_heads, dropout=0.1)
        
        # Position-wise feed-forward networks
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, 2, embed_dim)
        q = x[:,0,:].unsqueeze(0)  # (1, batch_size, embed_dim)
        k = x[:,1,:].unsqueeze(0)  # (1, batch_size, embed_dim)
        v = x[:,1,:].unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Multi-head attention
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.squeeze(0)  # (batch_size, embed_dim)
        
        # Add & Norm
        out1 = self.norm1(attn_output + x[:,0,:])
        
        # Feed Forward
        ff_output = self.feed_forward(out1)
        
        # Add & Norm
        out2 = self.norm2(ff_output + out1)
        
        # Classification
        return self.classifier(out2)

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=64, lr=1e-4):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_val_loss = float('inf')
    best_model = None
    patience = 20  # Increased patience
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training with {epochs} epochs...")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(torch.tensor(X_val, dtype=torch.float32)).squeeze()
            val_loss = criterion(val_out, torch.tensor(y_val, dtype=torch.float32))
            
            # Calculate validation accuracy
            val_preds = (val_out >= 0.5).float()
            val_acc = (val_preds == torch.tensor(y_val, dtype=torch.float32)).float().mean()

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch+1})")
            print("-" * 50)
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best model was from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
            break

    # Restore best model
    model.load_state_dict(best_model.state_dict())
    return best_val_loss, best_epoch


# def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=750, batch_size=32, lr=1e-4):
#     criterion = nn.BCELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Changed to AdamW with weight decay
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

#     # Convert to PyTorch datasets
#     train_dataset = torch.utils.data.TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.float32)
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True
#     )

#     best_val_loss = float('inf')
#     best_model = None
#     patience = 10
#     patience_counter = 0

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs.squeeze(), batch_y)
#             loss.backward()
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         # Validation
#         model.eval()
#         with torch.no_grad():
#             val_out = model(torch.tensor(X_val, dtype=torch.float32)).squeeze()
#             val_loss = criterion(val_out, torch.tensor(y_val, dtype=torch.float32))
        
#         # Learning rate scheduling
#         scheduler.step(val_loss)
        
#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model = copy.deepcopy(model)
#             patience_counter = 0
#         else:
#             patience_counter += 1
            
#         if patience_counter >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
            
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

#     # Restore best model
#     model.load_state_dict(best_model.state_dict())



# -------------------------------------------



def evaluate_model_sklearn(clf, X_test, y_test, model_name="Model"):
    # For non-PyTorch models like SVM and Logistic Regression
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds)
    print(f"{model_name} Accuracy:", acc)
    print(f"{model_name} Confusion Matrix:\n", cm)
    print(f"{model_name} Classification Report:\n", cr)

def evaluate_model_pytorch(model, X_test, y_test, model_name="NN Model"):
    # For PyTorch models like Attention NN or LSTM
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_t).squeeze()
        preds = (outputs >= 0.5).int().numpy()
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds)
    print(f"{model_name} Accuracy:", acc)
    print(f"{model_name} Confusion Matrix:\n", cm)
    print(f"{model_name} Classification Report:\n", cr)

# def main():
#     # 1. Gather data
#     data = gather_antonym_pairs()
#     sentences1 = [d[0] for d in data]
#     sentences2 = [d[1] for d in data]
#     labels = [d[2] for d in data]

#     # 2. Embed with nomic
#     model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
#     emb1 = model.encode(sentences1, show_progress_bar=False)
#     emb2 = model.encode(sentences2, show_progress_bar=False)
#     embeddings = np.stack([emb1, emb2], axis=1)

#     # 3. Train/test/val split
#     X_train, X_temp, y_train, y_temp = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     # 4. Polynomial SVM
#     svm_clf = SVC(kernel='poly', degree=2)
#     svm_clf.fit(X_train.reshape(len(X_train), -1), y_train)
#     print("Polynomial SVM score:", svm_clf.score(X_test.reshape(len(X_test), -1), y_test))

#     # 5. Simple Regression (Logistic Regression)
#     logreg = LogisticRegression()
#     logreg.fit(X_train.reshape(len(X_train), -1), y_train)
#     print("Logistic Regression score:", logreg.score(X_test.reshape(len(X_test), -1), y_test))

#     # 6. Single-Headed Attention NN
#     attn_model = SimpleAttentionNN(embed_dim=embeddings.shape[2])
#     train_pytorch_model(
#         attn_model,
#         X_train, y_train,
#         X_val, y_val,
#         epochs=5
#     )

#     # 7. LSTM Approach
#     lstm_model = LSTMBinaryClassifier(embed_dim=embeddings.shape[2], hidden_dim=128)
#     train_pytorch_model(
#         lstm_model,
#         X_train, y_train,
#         X_val, y_val,
#         epochs=5
#     )

#     # 8. Graph Transformer (Placeholder)
#     # This would require constructing a graph input (edge_index, etc.), beyond scope here.

#     print("Done training all models.")


def main():
    # 1. Gather data
    print("Gathering antonym pairs...")
    data = gather_antonym_pairs()
    print(f"Total antonym pairs: {len(data)}")
    sentences1 = [d[0] for d in data]
    sentences2 = [d[1] for d in data]
    labels = [d[2] for d in data]
    print("Data gathering complete.")

    # 2. Embed with nomic
    model_st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    # model_st = SentenceTransformer("microsoft/DiCE", device="cuda")
    print("Embedding sentences...")
    emb1 = model_st.encode(sentences1, show_progress_bar=False)
    emb2 = model_st.encode(sentences2, show_progress_bar=False)
    embeddings = np.stack([emb1, emb2], axis=1)
    print("Embedding complete.")

    # 3. Train/test/val split
    X_train, X_temp, y_train, y_temp = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("Data split complete.")
    # 4. Polynomial SVM
    # svm_clf = SVC(kernel='poly', degree=4,verbose=True)
    # svm_clf.fit(X_train.reshape(len(X_train), -1), y_train)
    # print("Polynomial SVM score:", svm_clf.score(X_test.reshape(len(X_test), -1), y_test))
    # evaluate_model_sklearn(svm_clf, X_test.reshape(len(X_test), -1), y_test, model_name="Polynomial SVM")

    # 5. Simple Regression (Logistic Regression)
    # logreg = LogisticRegression()
    # logreg.fit(X_train.reshape(len(X_train), -1), y_train)
    # print("Logistic Regression score:", logreg.score(X_test.reshape(len(X_test), -1), y_test))
    # evaluate_model_sklearn(logreg, X_test.reshape(len(X_test), -1), y_test, model_name="Logistic Regression")

    # 6. Single-Headed Attention NN
    # print(embeddings.shape)
    # attn_model = ImprovedAttentionNN(embed_dim=embeddings.shape[2])
    # train_pytorch_model(
    #     attn_model,
    #     X_train, y_train,
    #     X_val, y_val,
    #     epochs=1000,
    #     batch_size=64,
    #     lr=1e-4
    # )
    # evaluate_model_pytorch(attn_model, X_test, y_test, model_name="Attention NN")



    # 6. Improved Attention NN
    print("Training Improved Attention NN...")
    print(f"Input embedding dimension: {embeddings.shape[2]}")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print("-" * 50)
    
    attn_model = ImprovedAttentionNN(embed_dim=embeddings.shape[2])
    best_val_loss, best_epoch = train_pytorch_model(
        attn_model,
        X_train, y_train,
        X_val, y_val,
        epochs=1000,
        batch_size=64,
        lr=1e-4
    )
    
    print("\nFinal Evaluation:")
    print("-" * 50)
    evaluate_model_pytorch(attn_model, X_test, y_test, model_name="Improved Attention NN")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

    print("\nTraining complete!")

    # # 7. LSTM Approach
    # lstm_model = LSTMBinaryClassifier(embed_dim=embeddings.shape[2], hidden_dim=128)
    # train_pytorch_model(
    #     lstm_model,
    #     X_train, y_train,
    #     X_val, y_val,
    #     epochs=50
    # )
    # evaluate_model_pytorch(lstm_model, X_test, y_test, model_name="LSTM Model")

    # 8. Graph Transformer (Placeholder)
    # This would require constructing a graph input (edge_index, etc.), beyond scope here.

    print("Done training all models with extended analytics.")

if __name__ == "__main__":
    main()