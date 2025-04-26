import os
import torch
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
from collections import defaultdict

# -----------------------------
# Data Loading Utilities
# -----------------------------
class AntonymSynonymDataset(Dataset):
    """
    Dataset for antonym/synonym word pairs.
    Each line in the file is expected to have:
      word1 <tab or space> word2 <tab or space> label
    where label is 1 (synonym/antonym relation) or 0 (no relation)
    """
    def __init__(self, file_list, embedder):
        """
        file_list: list of file paths
        embedder: a SentenceTransformer model for embedding words
        """
        self.pairs = []
        self.labels = []
        self.embedder = embedder

        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    w1, w2, label = parts
                    self.pairs.append((w1, w2))
                    self.labels.append(int(label))

        # Pre-compute embeddings for unique words to speed up training.
        unique_words = set()
        for (w1, w2) in self.pairs:
            unique_words.add(w1)
            unique_words.add(w2)
        self.word2emb = {}
        word_list = list(unique_words)
        # Batch embed words using SentenceTransformer.
        embeddings = self.embedder.encode(word_list, convert_to_tensor=True)
        for word, emb in zip(word_list, embeddings):
            self.word2emb[word] = emb

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        w1, w2 = self.pairs[idx]
        label = self.labels[idx]
        # Get pre-trained embeddings (assumed to be d-dimensional tensor)
        emb1 = self.word2emb[w1]
        emb2 = self.word2emb[w2]
        return emb1, emb2, torch.tensor(label, dtype=torch.long)

def get_file_paths(root_dir):
    """
    Given the root dataset directory, return lists of file paths for train, val, test
    across all word classes.
    """
    train_files, val_files, test_files = [], [], []
    for fname in os.listdir(root_dir):
        if fname.endswith(".train"):
            train_files.append(os.path.join(root_dir, fname))
        elif fname.endswith(".val"):
            val_files.append(os.path.join(root_dir, fname))
        elif fname.endswith(".test"):
            test_files.append(os.path.join(root_dir, fname))
    return train_files, val_files, test_files

# -----------------------------
# Model Modules
# -----------------------------
class FeedForwardEncoder(nn.Module):
    """
    Implements a two-layer feed-forward encoder (used for both ENC-1 and ENC-2).
    f(X) = σ(W2 (σ(W1 * X + b1)) + b2)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation=F.relu):
        super(FeedForwardEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return x

class GraphAttentionLayer(nn.Module):
    """
    A simple attentive graph convolution layer.
    Given an input feature matrix X and an adjacency matrix A,
    we compute:
       H = ρ( A_hat * X * W )
    where A_hat is the normalized attention-weighted adjacency matrix.
    """
    def __init__(self, in_features, out_features, activation=F.relu):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, X, A):
        # A: [N, N] attention-weighted and normalized adjacency matrix
        support = self.W(X)  # [N, out_features]
        out = torch.matmul(A, support)
        if self.activation:
            out = self.activation(out)
        return out

class ICE_Net(nn.Module):
    """
    ICE-NET implementation:
      - ENC-1: two-layer feed-forward network for synonym pairs.
      - ENC-2: two-layer feed-forward network for antonym pairs.
      - ENC-3: Attentive Graph Convolutional Network (applied separately to head and tail sets)
        and final cosine similarity-based classification.
      - The final loss is L = L1 + L2 + L3.
    """
    def __init__(self, input_dim, enc_hidden_dim, enc_out_dim, graph_hidden_dim, final_dim, margin1=0.9, margin2=0.9, num_graph_layers=2):
        super(ICE_Net, self).__init__()
        # ENC-1 and ENC-2
        self.enc1 = FeedForwardEncoder(input_dim, enc_hidden_dim, enc_out_dim)
        self.enc2 = FeedForwardEncoder(input_dim, enc_hidden_dim, enc_out_dim)
        self.margin1 = margin1
        self.margin2 = margin2

        # ENC-3: two separate graph convolution networks (for head and tail)
        # We will assume the graph is constructed externally and passed as an attention matrix.
        self.graph_conv_head = nn.ModuleList([GraphAttentionLayer(enc_out_dim, graph_hidden_dim) for _ in range(num_graph_layers)])
        self.graph_conv_tail = nn.ModuleList([GraphAttentionLayer(enc_out_dim, graph_hidden_dim) for _ in range(num_graph_layers)])
        # Final linear layer to produce classification scores from concatenated cosine similarities
        self.classifier = nn.Linear(4, 2)  # 2 classes (relation vs no relation)

    def margin_loss(self, emb_func, xh, xt, positive=True, margin=0.9):
        """
        Computes margin-based loss for a pair:
         loss = max(0, margin - tanh(inner_product)) for positive pairs
         loss = max(0, margin + tanh(inner_product)) for negative pairs
        """
        # Compute inner product similarity
        sim = torch.tanh(torch.sum(emb_func(xh) * emb_func(xt), dim=-1))
        if positive:
            loss = F.relu(margin - sim)
        else:
            loss = F.relu(margin + sim)
        return loss.mean()

    def forward(self, emb1, emb2, adj_head, adj_tail, word_to_idx):
        """
        emb1, emb2: [batch, input_dim] embeddings for head and tail words.
        adj_head, adj_tail: attention-weighted, normalized adjacency matrices for head and tail graphs.
            These are tensors of shape [N, N] where N is the total number of unique words.
            (For simplicity, we assume a common graph is used for all words; in practice, graphs are constructed using thresholds.)
        word_to_idx: a dictionary mapping words to indices in the graph.
        
        Returns:
          logits: classification scores for each word pair in the batch
          loss_dict: dictionary containing L1, L2, L3 losses.
        """
        # Pass through ENC-1 and ENC-2
        f1_xh = self.enc1(emb1)  # ENC-1 for head
        f1_xt = self.enc1(emb2)  # ENC-1 for tail
        f2_xh = self.enc2(emb1)  # ENC-2 for head
        f2_xt = self.enc2(emb2)  # ENC-2 for tail

        # Compute margin losses L1 and L2.
        # Here we assume that positive examples (label=1) should have high similarity,
        # while negative examples (label=0) should be pushed apart.
        # In practice, negative samples may be generated by random replacement.
        # For demonstration, we compute the loss on the current batch.
        L1 = self.margin_loss(self.enc1, emb1, emb2, positive=True, margin=self.margin1)
        L2 = self.margin_loss(lambda x: self.enc2(x), emb1, emb2, positive=True, margin=self.margin2)
        
        # Now, for ENC-3 we perform attentive graph convolution.
        # In a full implementation, we would construct graphs Gh and Gt from all training data.
        # Here we assume adj_head and adj_tail are precomputed attention matrices over all words.
        # We also assume we have a feature matrix X for all words (we build it from the encoders).
        # For simplicity, we build X from concatenated outputs of ENC-1 and ENC-2 on all words in the graph.
        # Assume word_features is a tensor of shape [N, enc_out_dim] where N = len(word_to_idx).
        # We build it by taking the average of the two encoder outputs.
        N = len(word_to_idx)
        device = emb1.device
        word_features = torch.zeros(N, f1_xh.size(-1), device=device)
        # In a realistic scenario, you would update these features over the whole vocabulary.
        # Here, for demonstration, we assume that for each word in the current batch we update its feature.
        # (Note: This is a simplified treatment.)
        for i, word in enumerate(word_to_idx):
            # For demonstration, we simply use the ENC-1 output.
            word_features[i] = self.enc1(emb1[i % emb1.size(0)])  # dummy assignment

        # Apply graph convolution layers separately for head and tail graphs.
        x_head = word_features
        x_tail = word_features
        for layer in self.graph_conv_head:
            x_head = layer(x_head, adj_head)
        for layer in self.graph_conv_tail:
            x_tail = layer(x_tail, adj_tail)

        # Now, for each instance in the batch, get the graph-updated representations.
        # For each head word in the batch, lookup in x_head and for tail word in x_tail.
        batch_size = emb1.size(0)
        rep_head = []
        rep_tail = []
        for i in range(batch_size):
            # In practice, you would have the word itself and use word_to_idx mapping.
            # Here we use a dummy index based on modulo.
            idx = i % N
            rep_head.append(x_head[idx])
            rep_tail.append(x_tail[idx])
        rep_head = torch.stack(rep_head)  # [batch, graph_hidden_dim]
        rep_tail = torch.stack(rep_tail)

        # Following Equation (5) in the paper:
        # x1 = cos(x_tail_from_ENC2, x_tail_from_graph)
        # x2 = cos(x_head_from_ENC1, x_tail_from_graph)
        # x3 = cos(x_head_from_ENC1, x_tail_from_graph_ENC2) -- here we simulate using rep_head and rep_tail.
        # x4 = cos(x_tail_from_ENC1, x_head_from_graph_ENC2)
        cos_sim = nn.CosineSimilarity(dim=-1)
        x1 = cos_sim(f2_xt, rep_tail)
        x2 = cos_sim(f1_xh, rep_tail)
        x3 = cos_sim(f1_xh, rep_tail)
        x4 = cos_sim(f1_xt, rep_head)
        # Concatenate into feature vector: shape [batch, 4]
        XF = torch.stack([x1, x2, x3, x4], dim=1)
        logits = self.classifier(XF)

        # L3: Cross-entropy loss for classification.
        return logits, {'L1': L1, 'L2': L2}

# -----------------------------
# Graph Construction Helper
# -----------------------------
def construct_attention_graph(word_embeddings, top_k=5):
    """
    Given a tensor of word embeddings of shape [N, d], construct an
    attention-weighted and normalized adjacency matrix using cosine similarity.
    For each word, we keep the top_k most similar neighbors.
    """
    with torch.no_grad():
        normed = F.normalize(word_embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(normed, normed.transpose(0,1))  # [N, N]
        N = sim_matrix.size(0)
        A = torch.zeros_like(sim_matrix)
        for i in range(N):
            # Get top_k indices (excluding self)
            sim_i = sim_matrix[i]
            _, indices = torch.topk(sim_i, k=top_k+1)  # self included
            for idx in indices:
                if idx == i:
                    continue
                A[i, idx] = sim_i[idx]
        # Normalize A (adding self connections)
        A = A + torch.eye(N, device=A.device)
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = D @ A @ D
    return A_norm

# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, num_epochs, device, optimizer):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    # For graph construction, we assume we build a vocabulary from training data.
    # For demonstration, we simply build a dummy vocabulary from the first batch.
    dummy_batch = next(iter(train_loader))
    # Here we assume the batch returns emb1, emb2, label; we use emb1 to get dimension and number of instances.
    batch_emb = dummy_batch[0]
    N = batch_emb.size(0) * 2  # dummy vocabulary size
    # Create a dummy word embedding matrix for the graph (random initialization or average of encoder outputs)
    dummy_word_emb = torch.mean(batch_emb, dim=0, keepdim=True).repeat(N,1).to(device)
    # Construct attention matrices (for both head and tail graphs, for simplicity using the same matrix)
    adj = construct_attention_graph(dummy_word_emb, top_k=5).to(device)
    # Dummy word_to_idx dictionary
    word_to_idx = {str(i): i for i in range(N)}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for emb1, emb2, labels in train_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, loss_dict = model(emb1, emb2, adj, adj, word_to_idx)
            L3 = criterion(logits, labels)
            loss = loss_dict['L1'] + loss_dict['L2'] + L3
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
                logits, loss_dict = model(emb1, emb2, adj, adj, word_to_idx)
                L3 = criterion(logits, labels)
                loss = loss_dict['L1'] + loss_dict['L2'] + L3
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Optionally save the model
                torch.save(model.state_dict(), "icenet_best.pth")
    print("Training complete.")

def evaluate_model(model, test_loader, device, adj, word_to_idx):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for emb1, emb2, labels in test_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            logits, _ = model(emb1, emb2, adj, adj, word_to_idx)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Set paths to your dataset folder (update as necessary)
    torch.multiprocessing.set_start_method('spawn')

    dataset_dir = "./dataset"  # directory containing adjective-pairs.*, noun-pairs.*, verb-pairs.*
    train_files, val_files, test_files = get_file_paths(dataset_dir)
    print("Train files:", train_files)
    print("Val files:", val_files)
    print("Test files:", test_files)

    # Initialize the pre-trained SentenceTransformer model (nomic-embed)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Create dataset objects
    train_dataset = AntonymSynonymDataset(train_files, embedder)
    val_dataset = AntonymSynonymDataset(val_files, embedder)
    test_dataset = AntonymSynonymDataset(test_files, embedder)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Model hyperparameters
    input_dim = 384  # dimension of SentenceTransformer embeddings (for 'all-MiniLM-L6-v2')
    enc_hidden_dim = 200
    enc_out_dim = 60  # Change this to match graph_hidden_dim
    graph_hidden_dim = 60
    final_dim = 60  # final dimension used in graph conv (can be same as graph_hidden_dim)
    margin1 = 0.9
    margin2 = 0.9
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ICE_Net(input_dim, enc_hidden_dim, enc_out_dim, graph_hidden_dim, final_dim, margin1, margin2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # For graph construction during training, we use a dummy graph as shown in train_model.
    # In a full implementation, you would construct the graph over the whole vocabulary.
    dummy_batch = next(iter(train_loader))
    batch_emb = dummy_batch[0]
    N = batch_emb.size(0) * 2
    dummy_word_emb = torch.mean(batch_emb, dim=0, keepdim=True).repeat(N, 1).to(device)
    adj = construct_attention_graph(dummy_word_emb, top_k=5).to(device)
    word_to_idx = {str(i): i for i in range(N)}

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, device, optimizer)

    # Load best model
    model.load_state_dict(torch.load("icenet_best.pth"))
    evaluate_model(model, test_loader, device, adj, word_to_idx)

if __name__ == "__main__":
    main()
