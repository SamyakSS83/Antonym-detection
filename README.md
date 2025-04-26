# Antonym Synonym Detection

This repository contains implementations of various models for antonym detection, including graph-based models and dual encoder approaches.

## Repository Structure

- [`test_implementations/`](./test_implementations/): Contains experimental implementations
    - `icenet.py`: Implementation of ICE-NET, a graph-based model for antonym detection
    - `ICE-NET.py`: Faithful implementation 
    

- [`other_graphs/`](./other_graphs/): Contains four graph-based models:
    - **Graph Attention Network (GAT)**: Uses attention mechanisms to weigh neighbor features
    - **Graph Convolutional Network (GCN)**: Standard graph convolutions for learning node representations
    - **GraphSAGE**: Sample and aggregate approach for inductive learning on graphs
    - **Graph Isomorphism Network (GIN)**: More powerful than standard GCNs for learning graph structures

- [`multi_head/`](./multi_head/): Contains multi-head attention versions of graph models for better representation learning

### Core Models:
- `bert.py`: Base BERT model for word relation classification
- `dual_enc.py`: Dual encoder model with graph transformers
- Various diagnosis and evaluation files


### ICE-NET (`test_implementations`)
ICE-NET is a graph-based model for antonym detection with three key components:

1. **ENC-1**: Synonym Symmetry Encoder - projects word pairs into synonym space
2. **ENC-2**: Antonym Symmetry Encoder - projects word pairs into antonym space
3. **ENC-3**: Graph-based Transitivity Encoder - uses graph convolution for capturing transitive relations

The model constructs separate graphs for head and tail words and applies graph convolutions to capture higher-order relationships between word pairs.


## Dual Encoder Graph Model
This is the main recommended approach that combines BERT embeddings with graph-based learning.

#### Architecture:
1. **Dual Projection Branches**:
     - Synonym projection branch: `x_syn = Dropout(ReLU(W_syn·x + b_syn))`
     - Antonym projection branch: `x_ant = Dropout(ReLU(W_ant·x + b_ant))`

2. **Feature Fusion**:
     - Concatenation: `x_combined = [x_syn; x_ant]`
     - Linear transformation: `x_fused = W_f·x_combined + b_f`

3. **Graph Transformer Convolution**:
     - Multiple layers of transformer convolutions to capture graph structure

4. **Pooling and Classification**:
     - Global mean pooling followed by MLP for final classification

#### Loss Function:
The model uses a combination of binary cross-entropy loss and margin-based loss:
- For synonym pairs: push similarity in synonym space above margin
- For antonym pairs: push similarity in antonym space below margin

## Mathematical Formulation

### Problem Definition

Given a word pair dataset $\mathcal{W} = \{(w_i^1, w_i^2, y_i)\}_{i=1}^N$ with embeddings $\mathbf{x}_i^1$ and $\mathbf{x}_i^2$ for words $w_i^1$ and $w_i^2$, our goal is to learn a function $f: (\mathbf{x}_i^1, \mathbf{x}_i^2) \rightarrow \{0, 1\}$ that predicts the antonym relationship.

### Word Pair Graph Construction

For each word pair $(w_i^1, w_i^2)$, we construct a graph $G_i = (V_i, E_i)$ where:
- $V_i = \{v_i^1, v_i^2\}$ are nodes representing the words
- $E_i = \{(v_i^1, v_i^2), (v_i^2, v_i^1)\}$ are bidirectional edges
- Node features are embeddings: $\mathbf{X}_i = [\mathbf{x}_i^1, \mathbf{x}_i^2]^\top$

### Dual Projection Branches

The model projects input embeddings into synonym and antonym spaces:

$$\mathbf{x}_{syn} = \text{Dropout}(\text{ReLU}(W_{syn}\mathbf{x} + \mathbf{b}_{syn}))$$
$$\mathbf{x}_{ant} = \text{Dropout}(\text{ReLU}(W_{ant}\mathbf{x} + \mathbf{b}_{ant}))$$

where:
- $\mathbf{x} \in \mathbb{R}^d$ is the input node embedding
- $W_{syn}, W_{ant} \in \mathbb{R}^{h \times d}$ are projection matrices
- $\mathbf{b}_{syn}, \mathbf{b}_{ant} \in \mathbb{R}^h$ are bias vectors
- $\mathbf{x}_{syn}, \mathbf{x}_{ant} \in \mathbb{R}^h$ are projected representations

### Feature Fusion

The two branch outputs are concatenated and fused:

$$\mathbf{x}_{fused} = W_f[\mathbf{x}_{syn}; \mathbf{x}_{ant}] + \mathbf{b}_f$$

where $[\mathbf{x}_{syn}; \mathbf{x}_{ant}]$ is concatenation of both feature vectors.

### Graph Transformer Convolution

For a graph with node features and edge index:

$$\mathbf{X}^{(0)} = \mathbf{X}_{fused}$$
$$\mathbf{X}^{(l)} = \text{Dropout}(\text{ReLU}(\text{TransformerConv}(\mathbf{X}^{(l-1)}, E)))$$

The TransformerConv operation for node $i$ at layer $l$ is:

$$\mathbf{x}_i^{(l)} = \mathbf{W}_O^{(l)} \left[ \bigoplus_{h=1}^H \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{i,j}^{h,(l)} \mathbf{W}_V^{h,(l)} \mathbf{x}_j^{(l-1)} \right) \right]$$

where:
- $\mathcal{N}(i)$ is the neighborhood of node $i$
- $\alpha_{i,j}^{h,(l)}$ is the attention coefficient for head $h$ between nodes $i$ and $j$
- $\bigoplus$ denotes concatenation across heads

The attention coefficients are computed as:

$$\alpha_{i,j}^{h,(l)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{h,(l)\top}[\mathbf{W}_Q^{h,(l)}\mathbf{x}_i^{(l-1)} \parallel \mathbf{W}_K^{h,(l)}\mathbf{x}_j^{(l-1)}]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{h,(l)\top}[\mathbf{W}_Q^{h,(l)}\mathbf{x}_i^{(l-1)} \parallel \mathbf{W}_K^{h,(l)}\mathbf{x}_k^{(l-1)}]\right)\right)}$$

### Global Pooling and Classification

The final node representations are pooled:

$$\mathbf{x}_{pool} = \text{global\_mean\_pool}(\mathbf{X}^{(L)}) = \frac{1}{|V|}\sum_{i \in V} \mathbf{x}_i^{(L)}$$

This is followed by MLP for classification:

$$\hat{y} = \sigma(\mathbf{W}_2\text{Dropout}(\text{ReLU}(\mathbf{W}_1\mathbf{x}_{pool} + \mathbf{b}_1)) + \mathbf{b}_2)$$

### Loss Function

The model uses a combined loss function:

$$\mathcal{L} = \mathcal{L}_{BCE}(\hat{y}, y) + \lambda \mathcal{L}_{margin}$$

#### Binary Cross-Entropy Loss

$$\mathcal{L}_{BCE}(\hat{y}, y) = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

#### Margin-Based Loss

For a word pair $(w_1, w_2)$, the margin loss is:

$$\mathcal{L}_{margin} = 
\begin{cases}
\max(0, m_{syn} - \tanh(\langle \mathbf{x}_{syn}^1, \mathbf{x}_{syn}^2 \rangle)) & \text{if } y = 0 \text{ (synonym pair)} \\
\max(0, \tanh(\langle \mathbf{x}_{ant}^1, \mathbf{x}_{ant}^2 \rangle) - m_{ant}) & \text{if } y = 1 \text{ (antonym pair)}
\end{cases}$$

where:
- $\langle \cdot, \cdot \rangle$ denotes dot product similarity
- $m_{syn} = 0.8$ and $m_{ant} = 0.2$ are margin thresholds
- For synonym pairs: similarity in synonym space should exceed $m_{syn}$
- For antonym pairs: similarity in antonym space should be below $m_{ant}$

## How to Run

### Prerequisites
- Python 3.7+
- PyTorch
- PyTorch Geometric
- Transformers library
- Sentence-Transformers

### Training Process
1. First, train the base BERT model:
```bash
python dual_enc/bert.py
python dual_enc/bert.py
```
This will train the foundational BERT model and save embeddings for the next step.

2. Then, run the dual encoder model:
```bash
python dual_enc/dualenc_frombert.py
```
This will load the pre-trained BERT embeddings and train the dual encoder graph model.

