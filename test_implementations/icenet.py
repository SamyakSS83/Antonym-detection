import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
EMBEDDING_DIM = 384  # For sentence transformers
ENC1_DIM = 80
ENC2_DIM = 80
ENC3_DIM = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
GAMMA1 = 0.9
GAMMA2 = 0.9
REL_THR = 0.15
UNREL_THR = 0.10

class WordPairDataset(Dataset):
    def __init__(self, file_path, st_model=None, word_to_idx=None, embeddings=None):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word1 = parts[0]
                    word2 = parts[1]
                    label = int(parts[2])  # 1 for related, 0 for unrelated
                    self.data.append((word1, word2, label))
        
        self.st_model = st_model
        self.word_to_idx = word_to_idx
        self.embeddings = embeddings
        
        # Create word_to_idx if not provided
        if self.word_to_idx is None:
            self.create_word_index()
            
        # Create embeddings if not provided
        if self.embeddings is None and self.st_model is not None:
            self.create_embeddings()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word1, word2, label = self.data[idx]
        
        # Get word indices
        idx1 = self.word_to_idx.get(word1, 0)
        idx2 = self.word_to_idx.get(word2, 0)
        
        # Get embeddings
        emb1 = torch.tensor(self.embeddings[idx1], dtype=torch.float)
        emb2 = torch.tensor(self.embeddings[idx2], dtype=torch.float)
        
        return {
            'word1': word1,
            'word2': word2,
            'idx1': idx1,
            'idx2': idx2,
            'emb1': emb1,
            'emb2': emb2,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def create_word_index(self):
        words = set()
        for word1, word2, _ in self.data:
            words.add(word1)
            words.add(word2)
        
        self.word_to_idx = {word: i+1 for i, word in enumerate(words)}
        # Add padding index
        self.word_to_idx['<PAD>'] = 0
    
    def create_embeddings(self):
        words = ['<PAD>'] + list(self.word_to_idx.keys())
        self.embeddings = np.zeros((len(words), EMBEDDING_DIM))
        
        # Encode all words at once for efficiency
        word_embeddings = self.st_model.encode(words[1:])
        
        # Assign embeddings
        for i, emb in enumerate(word_embeddings):
            self.embeddings[i+1] = emb


class ENC1(nn.Module):
    """Synonym Symmetry Encoder"""
    def __init__(self, input_dim, output_dim):
        super(ENC1, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim * 2)
        self.layer2 = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x


class ENC2(nn.Module):
    """Antonym Symmetry Encoder"""
    def __init__(self, input_dim, output_dim):
        super(ENC2, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim * 2)
        self.layer2 = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x


class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ENC3(nn.Module):
    """Graph-based Transitivity Encoder"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ENC3, self).__init__()
        self.gc1_hh = GraphConvolution(input_dim, hidden_dim)
        self.gc2_hh = GraphConvolution(hidden_dim, output_dim)
        
        self.gc1_ht = GraphConvolution(input_dim, hidden_dim)
        self.gc2_ht = GraphConvolution(hidden_dim, output_dim)
        
        self.gc1_th = GraphConvolution(input_dim, hidden_dim)
        self.gc2_th = GraphConvolution(hidden_dim, output_dim)
        
        self.gc1_tt = GraphConvolution(input_dim, hidden_dim)
        self.gc2_tt = GraphConvolution(hidden_dim, output_dim)
    
    def forward(self, inputs, adj_h, adj_t):
        f1_h, f1_t, f2_h, f2_t = inputs
        
        # Process through graph convolutions
        Xhh = self.gc2_hh(F.relu(self.gc1_hh(f1_h, adj_h)), adj_h)
        Xht = self.gc2_ht(F.relu(self.gc1_ht(f1_t, adj_t)), adj_t)
        Xth = self.gc2_th(F.relu(self.gc1_th(f2_h, adj_h)), adj_h)
        Xtt = self.gc2_tt(F.relu(self.gc1_tt(f2_t, adj_t)), adj_t)
        
        return Xhh, Xht, Xth, Xtt


class ICENet(nn.Module):
    def __init__(self, embedding_dim, enc1_dim, enc2_dim, enc3_dim, num_classes=2):
        super(ICENet, self).__init__()
        
        # ENC-1: Synonym Symmetry Encoder
        self.enc1 = ENC1(embedding_dim, enc1_dim)
        
        # ENC-2: Antonym Symmetry Encoder
        self.enc2 = ENC2(embedding_dim, enc2_dim)
        
        # ENC-3: Graph-based Transitivity Encoder
        self.enc3 = ENC3(enc1_dim, enc3_dim * 2, enc3_dim)
        
        # Final classification layer
        self.classifier = nn.Linear(4, num_classes)
        
    def forward(self, h_emb, t_emb, adj_h=None, adj_t=None, h_idx=None, t_idx=None, all_nodes=None):
        # ENC-1 and ENC-2 outputs
        h_enc1 = self.enc1(h_emb)
        t_enc1 = self.enc1(t_emb)
        h_enc2 = self.enc2(h_emb)
        t_enc2 = self.enc2(t_emb)
        
        # For training without graph convolution (initial model)
        if adj_h is None or adj_t is None or all_nodes is None:
            # Compute scores directly
            x1 = F.cosine_similarity(h_enc2, t_enc2, dim=1).unsqueeze(1)
            x2 = F.cosine_similarity(h_enc1, t_enc1, dim=1).unsqueeze(1)
            x3 = F.cosine_similarity(h_enc1, t_enc2, dim=1).unsqueeze(1)
            x4 = F.cosine_similarity(t_enc1, h_enc2, dim=1).unsqueeze(1)
            
            # Concatenate scores
            scores = torch.cat([x1, x2, x3, x4], dim=1)
            
            # Final classification
            logits = self.classifier(scores)
            return logits, h_enc1, t_enc1, h_enc2, t_enc2
        
        # Process all nodes through ENC-1 and ENC-2
        all_h_enc1 = self.enc1(all_nodes)
        all_t_enc1 = self.enc1(all_nodes)
        all_h_enc2 = self.enc2(all_nodes)
        all_t_enc2 = self.enc2(all_nodes)
        
        # ENC-3: Graph-based processing with attentive graph convolutions
        # Apply graph convolution to get final representations
        Xhh, Xht, Xth, Xtt = self.enc3(
            (all_h_enc1, all_t_enc1, all_h_enc2, all_t_enc2), 
            adj_h, adj_t
        )
        
        # Get batch representations
        batch_Xhh = Xhh[h_idx]
        batch_Xht = Xht[t_idx]
        batch_Xth = Xth[h_idx]
        batch_Xtt = Xtt[t_idx]
        
        # Compute scores
        x1 = F.cosine_similarity(batch_Xth, batch_Xtt, dim=1).unsqueeze(1)
        x2 = F.cosine_similarity(batch_Xhh, batch_Xht, dim=1).unsqueeze(1)
        x3 = F.cosine_similarity(batch_Xhh, batch_Xtt, dim=1).unsqueeze(1)
        x4 = F.cosine_similarity(batch_Xht, batch_Xth, dim=1).unsqueeze(1)
        
        # Concatenate scores
        scores = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Final classification
        logits = self.classifier(scores)
        return logits, h_enc1, t_enc1, h_enc2, t_enc2
    
    def compute_loss(self, h_enc1, t_enc1, h_enc2, t_enc2, labels, neg_h_enc1=None, neg_t_enc1=None, neg_h_enc2=None, neg_t_enc2=None):
        # L1: Synonym symmetry loss
        pos_sim_syn = torch.tanh(torch.sum(h_enc1 * t_enc1, dim=1))
        L1_pos = torch.clamp(GAMMA1 - pos_sim_syn, min=0).mean()
        
        # Negative samples for L1
        if neg_h_enc1 is not None and neg_t_enc1 is not None:
            neg_sim_syn = torch.tanh(torch.sum(neg_h_enc1 * neg_t_enc1, dim=1))
            L1_neg = torch.clamp(GAMMA1 + neg_sim_syn, min=0).mean()
            L1 = L1_pos + L1_neg
        else:
            L1 = L1_pos
        
        # L2: Antonym symmetry loss
        pos_sim_ant = torch.tanh(torch.sum(h_enc2 * t_enc1, dim=1))
        L2_pos = torch.clamp(GAMMA2 - pos_sim_ant, min=0).mean()
        
        # Negative samples for L2
        if neg_h_enc2 is not None and neg_t_enc1 is not None:
            neg_sim_ant = torch.tanh(torch.sum(neg_h_enc2 * neg_t_enc1, dim=1))
            L2_neg = torch.clamp(GAMMA2 + neg_sim_ant, min=0).mean()
            L2 = L2_pos + L2_neg
        else:
            L2 = L2_pos
        
        return L1, L2


def create_graphs(model, dataset, device):
    """Create graphs for head and tail words."""
    # Initialize dictionaries to store related and unrelated pairs
    rel_h = {}
    unrel_h = {}
    rel_t = {}
    unrel_t = {}
    
    # Disable gradient computation
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE):
            h_emb = batch['emb1'].to(device)
            t_emb = batch['emb2'].to(device)
            
            # Forward pass to get scores
            logits, h_enc1, t_enc1, h_enc2, t_enc2 = model(h_emb, t_emb)
            
            # Get predicted probabilities
            probs = F.softmax(logits, dim=1)
            
            # Process each word pair
            for i in range(len(batch['word1'])):
                h_word = batch['word1'][i]
                t_word = batch['word2'][i]
                
                # Check if it's a probable related pair
                if probs[i, 1] >= REL_THR:  # Related probability
                    if h_word not in rel_h:
                        rel_h[h_word] = []
                    rel_h[h_word].append(t_word)
                    
                    if t_word not in rel_t:
                        rel_t[t_word] = []
                    rel_t[t_word].append(h_word)
                
                # Check if it's a probable unrelated pair
                elif probs[i, 0] >= UNREL_THR:  # Unrelated probability
                    if h_word not in unrel_h:
                        unrel_h[h_word] = []
                    unrel_h[h_word].append(t_word)
                    
                    if t_word not in unrel_t:
                        unrel_t[t_word] = []
                    unrel_t[t_word].append(h_word)
    
    # Create adjacency matrices
    all_words = list(set(list(rel_h.keys()) + list(unrel_h.keys()) + 
                      list(rel_t.keys()) + list(unrel_t.keys())))
    word_to_idx = {word: i for i, word in enumerate(all_words)}
    
    # Initialize adjacency matrices
    n = len(all_words)
    adj_h = torch.zeros((n, n), device=device)
    adj_t = torch.zeros((n, n), device=device)
    
    # Fill adjacency matrix for head words (Gh)
    for t_word in rel_t:
        if t_word in word_to_idx:
            # Create edges between head words that have the same tail
            heads = rel_t[t_word]
            for i in range(len(heads)):
                if heads[i] in word_to_idx:
                    for j in range(i+1, len(heads)):
                        if heads[j] in word_to_idx:
                            idx_i = word_to_idx[heads[i]]
                            idx_j = word_to_idx[heads[j]]
                            # Use similarity score as weight
                            adj_h[idx_i, idx_j] = 0.8
                            adj_h[idx_j, idx_i] = 0.8
    
    for t_word in unrel_t:
        if t_word in word_to_idx:
            # Create edges between head words that have the same tail
            heads = unrel_t[t_word]
            for i in range(len(heads)):
                if heads[i] in word_to_idx:
                    for j in range(i+1, len(heads)):
                        if heads[j] in word_to_idx:
                            idx_i = word_to_idx[heads[i]]
                            idx_j = word_to_idx[heads[j]]
                            # Use similarity score as weight
                            adj_h[idx_i, idx_j] = 0.6
                            adj_h[idx_j, idx_i] = 0.6
    
    # Fill adjacency matrix for tail words (Gt)
    for h_word in rel_h:
        if h_word in word_to_idx:
            tails = rel_h[h_word]
            for i in range(len(tails)):
                if tails[i] in word_to_idx:
                    for j in range(i+1, len(tails)):
                        if tails[j] in word_to_idx:
                            idx_i = word_to_idx[tails[i]]
                            idx_j = word_to_idx[tails[j]]
                            adj_t[idx_i, idx_j] = 0.8
                            adj_t[idx_j, idx_i] = 0.8
    
    for h_word in unrel_h:
        if h_word in word_to_idx:
            tails = unrel_h[h_word]
            for i in range(len(tails)):
                if tails[i] in word_to_idx:
                    for j in range(i+1, len(tails)):
                        if tails[j] in word_to_idx:
                            idx_i = word_to_idx[tails[i]]
                            idx_j = word_to_idx[tails[j]]
                            adj_t[idx_i, idx_j] = 0.6
                            adj_t[idx_j, idx_i] = 0.6
    
    # Add self-loops and normalize
    adj_h = adj_h + torch.eye(n, device=device)
    adj_t = adj_t + torch.eye(n, device=device)
    
    # Normalize adjacency matrices
    d_h = torch.sum(adj_h, dim=1)
    d_t = torch.sum(adj_t, dim=1)
    
    d_h_inv_sqrt = torch.pow(d_h, -0.5)
    d_t_inv_sqrt = torch.pow(d_t, -0.5)
    
    d_h_inv_sqrt[torch.isinf(d_h_inv_sqrt)] = 0
    d_t_inv_sqrt[torch.isinf(d_t_inv_sqrt)] = 0
    
    d_h_inv_sqrt = torch.diag(d_h_inv_sqrt)
    d_t_inv_sqrt = torch.diag(d_t_inv_sqrt)
    
    adj_h_normalized = torch.mm(torch.mm(d_h_inv_sqrt, adj_h), d_h_inv_sqrt)
    adj_t_normalized = torch.mm(torch.mm(d_t_inv_sqrt, adj_t), d_t_inv_sqrt)
    
    # Create embedding matrix for all words
    all_nodes = torch.zeros((n, EMBEDDING_DIM), device=device)
    for word, idx in word_to_idx.items():
        # Get word embedding from dataset
        for batch in DataLoader(dataset, batch_size=1):
            if batch['word1'][0] == word:
                all_nodes[idx] = batch['emb1'][0]
                break
            elif batch['word2'][0] == word:
                all_nodes[idx] = batch['emb2'][0]
                break
    
    return adj_h_normalized, adj_t_normalized, all_nodes, word_to_idx

def train_initial_model(model, train_loader, val_loader, device, epochs=10):
    """Train the initial model without graph convolutions."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            h_emb = batch['emb1'].to(device)
            t_emb = batch['emb2'].to(device)
            labels = batch['label'].to(device)
            
            # Create negative samples by shuffling
            neg_indices = torch.randperm(h_emb.size(0))
            neg_h_emb = h_emb[neg_indices]
            neg_t_emb = t_emb[neg_indices]
            
            # Forward pass
            logits, h_enc1, t_enc1, h_enc2, t_enc2 = model(h_emb, t_emb)
            _, neg_h_enc1, neg_t_enc1, neg_h_enc2, neg_t_enc2 = model(neg_h_emb, neg_t_emb)
            
            # Compute losses
            L1, L2 = model.compute_loss(h_enc1, t_enc1, h_enc2, t_enc2, labels, 
                                      neg_h_enc1, neg_t_enc1, neg_h_enc2, neg_t_enc2)
            L3 = criterion(logits, labels)
            
            # Total loss
            loss = L1 + L2 + L3
            total_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def train_full_model(model, train_loader, val_loader, device, adj_h, adj_t, all_nodes, word_to_idx, epochs=EPOCHS):
    """Train the full ICE-NET model with graph convolutions."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            h_emb = batch['emb1'].to(device)
            t_emb = batch['emb2'].to(device)
            labels = batch['label'].to(device)
            
            # Get word indices for graph lookup
            h_idx = torch.tensor([word_to_idx.get(word, 0) for word in batch['word1']], device=device)
            t_idx = torch.tensor([word_to_idx.get(word, 0) for word in batch['word2']], device=device)
            
            # Forward pass with graph convolutions
            logits, h_enc1, t_enc1, h_enc2, t_enc2 = model(h_emb, t_emb, adj_h, adj_t, h_idx, t_idx, all_nodes)
            
            # Compute loss
            L3 = criterion(logits, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            L3.backward()
            optimizer.step()
            
            total_loss += L3.item()
        
        # Validation
        val_metrics = evaluate_with_graph(model, val_loader, device, adj_h, adj_t, all_nodes, word_to_idx)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate(model, data_loader, device):
    """Evaluate model without graph convolutions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            h_emb = batch['emb1'].to(device)
            t_emb = batch['emb2'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _, _, _, _ = model(h_emb, t_emb)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_with_graph(model, data_loader, device, adj_h, adj_t, all_nodes, word_to_idx):
    """Evaluate model with graph convolutions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            h_emb = batch['emb1'].to(device)
            t_emb = batch['emb2'].to(device)
            labels = batch['label'].to(device)
            
            # Get word indices for graph lookup
            h_idx = torch.tensor([word_to_idx.get(word, 0) for word in batch['word1']], device=device)
            t_idx = torch.tensor([word_to_idx.get(word, 0) for word in batch['word2']], device=device)
            
            # Forward pass with graph convolutions
            logits, _, _, _, _ = model(h_emb, t_emb, adj_h, adj_t, h_idx, t_idx, all_nodes)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Sentence Transformer model
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load datasets
    dataset_dir = "dataset"
    
    # Load and process datasets
    adj_train = WordPairDataset(os.path.join(dataset_dir, "adjective-pairs.train"), st_model)
    adj_val = WordPairDataset(os.path.join(dataset_dir, "adjective-pairs.val"), st_model, 
                             adj_train.word_to_idx, adj_train.embeddings)
    adj_test = WordPairDataset(os.path.join(dataset_dir, "adjective-pairs.test"), st_model, 
                              adj_train.word_to_idx, adj_train.embeddings)
    
    noun_train = WordPairDataset(os.path.join(dataset_dir, "noun-pairs.train"), st_model)
    noun_val = WordPairDataset(os.path.join(dataset_dir, "noun-pairs.val"), st_model, 
                              noun_train.word_to_idx, noun_train.embeddings)
    noun_test = WordPairDataset(os.path.join(dataset_dir, "noun-pairs.test"), st_model, 
                               noun_train.word_to_idx, noun_train.embeddings)
    
    verb_train = WordPairDataset(os.path.join(dataset_dir, "verb-pairs.train"), st_model)
    verb_val = WordPairDataset(os.path.join(dataset_dir, "verb-pairs.val"), st_model, 
                              verb_train.word_to_idx, verb_train.embeddings)
    verb_test = WordPairDataset(os.path.join(dataset_dir, "verb-pairs.test"), st_model, 
                               verb_train.word_to_idx, verb_train.embeddings)
    
    # Create data loaders
    adj_train_loader = DataLoader(adj_train, batch_size=BATCH_SIZE, shuffle=True)
    adj_val_loader = DataLoader(adj_val, batch_size=BATCH_SIZE)
    adj_test_loader = DataLoader(adj_test, batch_size=BATCH_SIZE)
    
    noun_train_loader = DataLoader(noun_train, batch_size=BATCH_SIZE, shuffle=True)
    noun_val_loader = DataLoader(noun_val, batch_size=BATCH_SIZE)
    noun_test_loader = DataLoader(noun_test, batch_size=BATCH_SIZE)
    
    verb_train_loader = DataLoader(verb_train, batch_size=BATCH_SIZE, shuffle=True)
    verb_val_loader = DataLoader(verb_val, batch_size=BATCH_SIZE)
    verb_test_loader = DataLoader(verb_test, batch_size=BATCH_SIZE)
    
    # Train on adjectives
    print("\nTraining on adjective pairs...")
    adj_model = ICENet(EMBEDDING_DIM, ENC1_DIM, ENC2_DIM, ENC3_DIM).to(device)
    
    # Train initial model
    print("Training initial model...")
    adj_model = train_initial_model(adj_model, adj_train_loader, adj_val_loader, device)
    
    # Create graphs
    print("Creating graphs...")
    adj_h, adj_t, all_nodes, word_to_idx = create_graphs(adj_model, adj_train, device)
    
    # Train full model
    print("Training full model...")
    adj_model = train_full_model(adj_model, adj_train_loader, adj_val_loader, device, 
                                adj_h, adj_t, all_nodes, word_to_idx)
    
    # Evaluate
    adj_metrics = evaluate_with_graph(adj_model, adj_test_loader, device, adj_h, adj_t, all_nodes, word_to_idx)
    print(f"Adjective Test Results: Precision={adj_metrics['precision']:.4f}, Recall={adj_metrics['recall']:.4f}, F1={adj_metrics['f1']:.4f}")
    
    # Train on nouns
    print("\nTraining on noun pairs...")
    noun_model = ICENet(EMBEDDING_DIM, ENC1_DIM, ENC2_DIM, ENC3_DIM).to(device)
    
    # Train initial model
    print("Training initial model...")
    noun_model = train_initial_model(noun_model, noun_train_loader, noun_val_loader, device)
    
    # Create graphs
    print("Creating graphs...")
    noun_h, noun_t, noun_nodes, noun_word_to_idx = create_graphs(noun_model, noun_train, device)
    
    # Train full model
    print("Training full model...")
    noun_model = train_full_model(noun_model, noun_train_loader, noun_val_loader, device, 
                                 noun_h, noun_t, noun_nodes, noun_word_to_idx)
    
    # Evaluate
    noun_metrics = evaluate_with_graph(noun_model, noun_test_loader, device, noun_h, noun_t, noun_nodes, noun_word_to_idx)
    print(f"Noun Test Results: Precision={noun_metrics['precision']:.4f}, Recall={noun_metrics['recall']:.4f}, F1={noun_metrics['f1']:.4f}")
    
    # Train on verbs
    print("\nTraining on verb pairs...")
    verb_model = ICENet(EMBEDDING_DIM, ENC1_DIM, ENC2_DIM, ENC3_DIM).to(device)
    
    # Train initial model
    print("Training initial model...")
    verb_model = train_initial_model(verb_model, verb_train_loader, verb_val_loader, device)
    
    # Create graphs
    print("Creating graphs...")
    verb_h, verb_t, verb_nodes, verb_word_to_idx = create_graphs(verb_model, verb_train, device)
    
    # Train full model
    print("Training full model...")
    verb_model = train_full_model(verb_model, verb_train_loader, verb_val_loader, device, 
                                 verb_h, verb_t, verb_nodes, verb_word_to_idx)
    
    # Evaluate
    verb_metrics = evaluate_with_graph(verb_model, verb_test_loader, device, verb_h, verb_t, verb_nodes, verb_word_to_idx)
    print(f"Verb Test Results: Precision={verb_metrics['precision']:.4f}, Recall={verb_metrics['recall']:.4f}, F1={verb_metrics['f1']:.4f}")
    
    # Print summary of results
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    print(f"Adjective Pairs: Precision={adj_metrics['precision']:.4f}, Recall={adj_metrics['recall']:.4f}, F1={adj_metrics['f1']:.4f}")
    print(f"Noun Pairs: Precision={noun_metrics['precision']:.4f}, Recall={noun_metrics['recall']:.4f}, F1={noun_metrics['f1']:.4f}")
    print(f"Verb Pairs: Precision={verb_metrics['precision']:.4f}, Recall={verb_metrics['recall']:.4f}, F1={verb_metrics['f1']:.4f}")
    
    # Save models
    torch.save(adj_model.state_dict(), "adj_model.pt")
    torch.save(noun_model.state_dict(), "noun_model.pt")
    torch.save(verb_model.state_dict(), "verb_model.pt")

if __name__ == "__main__":
    main()
