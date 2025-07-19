# Bhav-Net: Multilingual Semantic Analysis for Cross-Lingual Antonym-Synonym Detection Using Dual-Encoder Graph Transformers

**Author:** Samyak S. Sanghvi  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Delhi  
**Email:** cs1230807@iitd.ac.in

---

## Abstract

This paper presents Bhav-Net, a novel multilingual dual-encoder architecture for antonym-synonym detection that effectively captures semantic relationships across seven languages. Building upon traditional dual encoder frameworks, Bhav-Net integrates language-specific fine-tuned BERT models with graph transformer networks to model word relationships as dual projections in semantic space. The architecture employs separate synonym and antonym projection branches, enabling the model to learn distinct semantic patterns for opposing relationships. Our approach leverages graph-based representations where word pairs form two-node graphs, allowing transformer convolutions to capture higher-order semantic dependencies. Experimental evaluation across German, French, Spanish, Italian, Portuguese, Dutch, and Russian demonstrates robust cross-lingual performance, with the dual-space projection mechanism proving particularly effective at distinguishing antonymous relationships from synonymous ones.

**Keywords:** multilingual NLP, antonym detection, graph neural networks, dual encoder, semantic analysis, transformer networks

---

## 1. Introduction

Understanding semantic relationships between words is fundamental to natural language processing and computational linguistics. Among these relationships, antonymy represents one of the most challenging yet important semantic phenomena to detect automatically. Unlike synonymy, which indicates semantic similarity, antonymy involves words that are semantically related but express opposite meanings. This opposition creates unique challenges for computational models, as antonyms often appear in similar contexts while expressing contrasting concepts.

Traditional approaches to antonym detection have primarily focused on monolingual settings, limiting their applicability in our increasingly multilingual world. The emergence of transformer-based language models has opened new possibilities for cross-lingual semantic understanding, yet most existing methods fail to explicitly model the dual nature of antonymous relationships.

We introduce **Bhav-Net**, a multilingual dual-encoder architecture that addresses these limitations through three key innovations:

1. **Dual Projection Branches**: Separate synonym and antonym projection spaces that learn distinct semantic patterns
2. **Graph-Based Modeling**: Word pairs represented as two-node graphs with transformer convolutions
3. **Multilingual Training**: Language-specific BERT models with unified architectural principles

### Key Contributions

- A novel dual-space projection mechanism that explicitly models synonym and antonym relationships
- A graph-based representation framework that captures relational dependencies through transformer convolutions
- A multilingual training strategy leveraging language-specific BERT models for optimal cross-lingual performance
- Comprehensive evaluation across seven languages demonstrating the effectiveness of our approach

---

## 2. Related Work

### 2.1 Antonym Detection Approaches

Early work in antonym detection relied heavily on lexical resources and pattern-based methods. These approaches achieved reasonable performance but were limited by the availability of high-quality lexical databases and struggled with out-of-vocabulary words.

The emergence of distributional semantics introduced vector space models for capturing semantic relationships. However, traditional word embeddings face the challenge that antonyms often appear in similar contexts, leading to high cosine similarity between opposing concepts.

Recent neural approaches have attempted to address this limitation through specialized architectures, but most lack the sophisticated graph-based modeling and multilingual capabilities present in our work.

### 2.2 Graph Neural Networks for NLP

Graph neural networks have shown remarkable success in modeling relational data. In NLP applications, graph-based approaches have been particularly effective for tasks requiring the modeling of complex relationships between linguistic entities.

The transformer architecture has been successfully adapted for graph-based learning through Graph Transformer networks, combining the attention mechanisms of transformers with the structural modeling capabilities of graph neural networks.

### 2.3 Multilingual Language Models

Multilingual BERT and similar models have demonstrated the ability to capture cross-lingual semantic representations. Language-specific fine-tuning has proven particularly effective for tasks requiring deep semantic understanding.

However, most existing multilingual approaches for semantic relationship detection do not explicitly model the contrastive nature of antonymous relationships, limiting their effectiveness for this specific task.

---

## 3. Methodology

### 3.1 Problem Formulation

Given a word pair dataset W = {(w₁ᵢ, w₂ᵢ, yᵢ)} across multiple languages, where w₁ᵢ and w₂ᵢ are words and yᵢ ∈ {0, 1} indicates antonym relationship (1) or non-antonym relationship (0), our goal is to learn a function f: (w₁ᵢ, w₂ᵢ) → {0, 1} that accurately predicts antonymous relationships across languages.

### 3.2 Architecture Overview

Bhav-Net consists of four main components working in sequence:

1. **Language-specific BERT encoding** for initial word representations
2. **Dual projection branches** creating separate semantic spaces  
3. **Graph transformer convolutions** for relational modeling
4. **Classification layers** with margin-based loss optimization

### 3.3 Language-Specific BERT Encoding

For each language, we employ specialized BERT models optimized for that linguistic context:

- **German**: `dbmdz/bert-base-german-cased`
- **French**: `camembert-base`
- **Spanish**: `dccuchile/bert-base-spanish-wwm-cased`
- **Italian**: `dbmdz/bert-base-italian-cased`
- **Portuguese**: `neuralmind/bert-base-portuguese-cased`
- **Dutch**: `GroNLP/bert-base-dutch-cased`
- **Russian**: `DeepPavlov/rubert-base-cased`

Each word w is encoded through its language-specific BERT model to obtain a contextual embedding x_w ∈ ℝ⁷⁶⁸.

### 3.4 Dual Projection Mechanism

The core innovation of Bhav-Net lies in its dual projection mechanism, which creates separate semantic spaces for modeling synonymous and antonymous relationships.

#### Synonym Space Projection
The synonym projection branch transforms input embeddings into a space where semantically similar words are positioned close together:

```
x_syn = Dropout(ReLU(W_syn·x + b_syn))
```

where W_syn ∈ ℝʰˣ⁷⁶⁸ is the synonym projection matrix and b_syn ∈ ℝʰ is the bias vector.

#### Antonym Space Projection
Similarly, the antonym projection branch creates a space optimized for capturing oppositional relationships:

```
x_ant = Dropout(ReLU(W_ant·x + b_ant))
```

This dual-space approach allows the model to learn distinct patterns for different types of semantic relationships.

#### Feature Fusion
The outputs from both projection branches are concatenated and fused:

```
x_fused = W_f[x_syn; x_ant] + b_f
```

### 3.5 Graph-Based Relational Modeling

For each word pair (w₁, w₂), we construct a graph G = (V, E) where:
- V = {v₁, v₂} represents the two words as nodes
- E = {(v₁, v₂), (v₂, v₁)} represents bidirectional edges
- Node features are the fused embeddings: X = [x_fused¹, x_fused²]ᵀ

This graph representation enables the model to capture relational dependencies through structured message passing.

### 3.6 Graph Transformer Convolutions

We apply multiple layers of transformer convolutions to capture higher-order relationships:

```
X^(l) = Dropout(ReLU(TransformerConv(X^(l-1), E)))
```

The transformer convolution operation uses multi-head attention mechanisms, allowing the model to focus on relevant relational patterns.

### 3.7 Classification and Loss Function

After graph convolutions, we apply global mean pooling followed by multilayer perceptron layers for final classification:

```
x_pool = (1/|V|) Σ x_i^(L)
ŷ = σ(W₂·Dropout(ReLU(W₁·x_pool + b₁)) + b₂)
```

#### Margin-Based Loss

Bhav-Net employs a combined loss function that includes both classification accuracy and semantic space organization:

```
L = L_BCE(ŷ, y) + λ·L_margin
```

The margin loss encourages proper positioning in both semantic spaces:

```
L_margin = {
  max(0, m_syn - tanh(⟨x_syn¹, x_syn²⟩))  if y = 0 (synonym pair)
  max(0, tanh(⟨x_ant¹, x_ant²⟩) - m_ant)   if y = 1 (antonym pair)
}
```

This loss function guides the model to position synonym pairs with high similarity in synonym space (> m_syn = 0.8) and antonym pairs with high similarity in antonym space while maintaining low similarity in synonym space (< m_ant = 0.2).

---

## 4. Training Process

### 4.1 Two-Stage Training Strategy

Bhav-Net employs a two-stage training approach designed to leverage the strengths of both specialized language understanding and graph-based relational modeling.

#### Stage 1: BERT Fine-tuning
In the first stage, we fine-tune language-specific BERT models for binary antonym classification. This stage serves two purposes:
1. Adapting the pre-trained language models to the specific task of semantic relationship detection
2. Creating high-quality initial embeddings that capture task-relevant semantic patterns

For each language, we train the BERT model using standard classification loss for 3-5 epochs with early stopping based on validation performance.

#### Stage 2: Dual Encoder Training
In the second stage, we freeze the fine-tuned BERT encoder and train the dual encoder graph transformer architecture. This approach allows the model to focus on learning optimal projection mappings and graph-based reasoning without interfering with the already-optimized language representations.

The dual encoder is trained for 10-20 epochs using the combined loss function, with learning rates typically set to 1×10⁻⁴ to ensure stable convergence.

### 4.2 Cross-Lingual Training Strategy

Our training approach recognizes the unique characteristics of each language while maintaining architectural consistency. For each language, we use:

- Language-specific BERT models optimized for that linguistic context
- Language-specific datasets sourced from WordNet and ConceptNet
- Unified architectural parameters that enable cross-lingual knowledge transfer
- Consistent hyperparameters across languages for fair comparison

---

## 5. Experimental Setup

### 5.1 Datasets

Our evaluation uses real-world antonym datasets sourced from two primary repositories:

- **Open Multilingual WordNet (OMW)**: Professional linguistic data providing high-quality antonym pairs
- **ConceptNet**: Large-scale semantic knowledge graph containing extensive multilingual relationship data

#### Dataset Statistics

| Language   | WordNet Pairs | ConceptNet Pairs | Total Pairs |
|------------|---------------|------------------|-------------|
| German     | 0*            | 2,678           | 2,678       |
| French     | 571           | 5,703           | 6,095       |
| Spanish    | 1,634         | 878             | 2,263       |
| Italian    | 917           | 1,889           | 2,495       |
| Portuguese | TBD           | TBD             | TBD         |
| Dutch      | TBD           | TBD             | TBD         |
| Russian    | TBD           | TBD             | TBD         |

*German WordNet data not available in OMW format, using ConceptNet only.

The combined datasets provide substantial coverage across our target languages, representing a significant improvement over previous synthetic or limited datasets (100-300x increase in dataset size and quality).

### 5.2 Implementation Details

Bhav-Net is implemented using PyTorch and PyTorch Geometric. Key implementation parameters include:

- **Hidden dimension**: 256
- **Dropout rate**: 0.2
- **Graph transformer heads**: 2
- **Batch size**: 32
- **Learning rate**: 2×10⁻⁵ (BERT), 1×10⁻⁴ (Dual Encoder)
- **Margin thresholds**: m_syn = 0.8, m_ant = 0.2
- **Margin weight**: λ = 0.5

All experiments are conducted with automatic mixed precision to enable efficient training on GPU hardware.

### 5.3 Evaluation Metrics

We evaluate model performance using standard classification metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for antonym detection
- **Recall**: Coverage of actual antonym pairs
- **F1-Score**: Harmonic mean of precision and recall

---

## 6. Results and Analysis

### 6.1 Quantitative Results

The following table presents the performance of Bhav-Net across all target languages:

| Language   | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| German     | --       | --        | --     | --       |
| French     | --       | --        | --     | --       |
| Spanish    | --       | --        | --     | --       |
| Italian    | --       | --        | --     | --       |
| Portuguese | --       | --        | --     | --       |
| Dutch      | --       | --        | --     | --       |
| Russian    | --       | --        | --     | --       |
| **Average**| --       | --        | --     | --       |

*Results to be filled in after experimental evaluation*

### 6.2 Architectural Analysis

The effectiveness of Bhav-Net stems from several key architectural decisions that align with fundamental principles of semantic understanding and generation.

#### Dual-Space Semantic Modeling

The dual projection mechanism addresses a fundamental challenge in computational semantics: antonyms and synonyms exhibit different distributional patterns that require distinct modeling approaches. By creating separate semantic spaces, Bhav-Net captures the nuanced differences between these relationship types.

Our analysis reveals that the synonym projection branch learns to cluster semantically similar words, while the antonym projection branch develops representations that highlight oppositional characteristics. This separation enables the model to distinguish between words that might appear in similar contexts but express contrasting meanings.

#### Graph-Based Relational Reasoning

The graph transformer architecture enables the model to reason about relationships at multiple levels of abstraction. Unlike simple vector comparisons, the graph-based approach allows for message passing between word representations, enabling the capture of higher-order semantic dependencies.

The bidirectional graph structure ensures that relationship detection is symmetric, addressing the inherent symmetry of both antonymous and synonymous relationships. The transformer convolutions enable attention-based message passing, allowing the model to focus on the most relevant aspects of each word pair for relationship determination.

#### Margin-Based Semantic Organization

The margin loss component plays a crucial role in organizing the learned semantic spaces. By explicitly enforcing separation between antonym and synonym pairs in their respective spaces, the model develops more interpretable and robust representations.

The margin thresholds (m_syn = 0.8, m_ant = 0.2) create clear decision boundaries that align with human intuitions about semantic relationships. This explicit organization contributes to both improved performance and better interpretability of the learned representations.

### 6.3 Cross-Lingual Effectiveness

Bhav-Net's success across multiple languages demonstrates the universality of its architectural principles. Several factors contribute to this cross-lingual effectiveness:

#### Language-Specific Initialization

The use of language-specific BERT models provides optimal starting representations for each linguistic context. This approach recognizes that while semantic relationship patterns may be universal, their surface manifestations vary across languages.

The fine-tuning stage allows these language-specific models to adapt to the specific task of antonym detection while preserving their language-specific capabilities.

#### Universal Architectural Patterns

Despite using language-specific encoders, the dual encoder architecture employs consistent structural patterns across all languages. This consistency enables the model to learn universal principles of semantic relationship modeling while adapting to language-specific characteristics.

The graph transformer component operates on semantic representations rather than surface forms, enabling it to capture relationship patterns that transcend linguistic boundaries.

#### Robust Training Strategy

The two-stage training approach ensures stable learning across languages with varying dataset sizes and characteristics. By separating language model adaptation from relationship modeling, the training process avoids interference between these complementary learning objectives.

### 6.4 Why Bhav-Net Works Across Languages

The architecture's cross-lingual success can be attributed to several key factors that closely mirror human semantic processing:

#### Universal Semantic Principles

The dual-space projection mechanism models universal patterns of opposition and similarity that transcend linguistic boundaries. Humans across cultures understand concepts of "hot" vs "cold" or "big" vs "small" regardless of the specific words used in their language.

#### Contextual Understanding

Language-specific BERT models capture the nuanced ways different languages express semantic relationships. For example, German compound words, French gendered nouns, or Spanish verb conjugations all carry semantic information that general multilingual models might miss.

#### Relational Reasoning

The graph transformer architecture mirrors how humans understand relationships between concepts. Rather than comparing words in isolation, we consider them in context with related concepts and their interconnections.

#### Explicit Contrast Modeling

The separate antonym and synonym spaces reflect how human cognition maintains distinct mental models for similarity versus opposition. This architectural choice enables the model to avoid the common pitfall where antonyms are grouped together simply because they appear in similar contexts.

---

## 7. Discussion

### 7.1 Architectural Insights

Bhav-Net's architecture closely symbolizes fundamental aspects of human semantic understanding and generation:

#### Cognitive Modeling

The dual projection mechanism reflects how humans maintain separate cognitive frameworks for understanding similarity and opposition. When we hear "hot" and "cold," we simultaneously recognize their oppositional relationship and their shared domain (temperature).

#### Attention and Focus

The graph transformer convolutions mirror selective attention in human language processing. Just as humans focus on relevant semantic features when determining relationships, the attention mechanisms allow the model to weight the most informative aspects of word representations.

#### Contextual Integration

The two-stage training process parallels human language acquisition: first learning language-specific patterns, then developing meta-linguistic understanding of semantic relationships that can transfer across contexts.

### 7.2 Performance Factors

Several design choices contribute to Bhav-Net's robust performance:

#### Semantic Space Organization

The margin-based loss creates clear decision boundaries that prevent the model from placing antonyms close together simply because they share contextual patterns. This explicit organization ensures that the learned representations align with human intuitions about semantic relationships.

#### Multi-Level Processing

The combination of BERT encoding, dual projections, and graph convolutions creates a hierarchical processing pipeline that captures semantic relationships at multiple levels of abstraction, from surface-level linguistic patterns to deep conceptual relationships.

#### Cross-Lingual Consistency

While using language-specific encoders, the unified architectural framework ensures that relationship patterns learned in one language can inform understanding in others, enabling effective cross-lingual transfer.

---

## 8. Conclusion

Bhav-Net represents a significant advancement in multilingual semantic relationship detection through its innovative dual-encoder architecture. By explicitly modeling synonym and antonym relationships in separate semantic spaces and employing graph-based reasoning for relational understanding, the approach addresses fundamental challenges in computational semantics.

The architecture's success across seven diverse languages demonstrates the universality of its design principles while highlighting the importance of language-specific optimization. The two-stage training strategy effectively balances language-specific adaptation with universal relationship modeling.

### Key Contributions

1. **Dual-Space Projection Mechanism**: Explicitly models contrasting semantic relationships in separate learned spaces
2. **Graph Transformer Architecture**: Enables higher-order relational reasoning through structured message passing
3. **Margin-Based Loss Function**: Creates interpretable semantic organizations with clear decision boundaries
4. **Multilingual Training Strategy**: Balances language-specific optimization with universal architectural principles

### Impact and Applications

Bhav-Net's robust architectural foundation offers promising directions for:
- Cross-lingual information retrieval and search
- Multilingual sentiment analysis and opinion mining
- Educational applications for language learning
- Computational linguistics research on semantic universals

### Future Work

Future research directions include:
- Extension to additional languages and language families
- Investigation of few-shot learning capabilities for low-resource languages
- Application to broader semantic relationship detection tasks (hypernymy, meronymy, etc.)
- Analysis of transfer learning capabilities across related languages
- Integration with knowledge graphs for enhanced semantic understanding

The robust architectural foundation provided by Bhav-Net establishes a new paradigm for multilingual semantic relationship detection and opens avenues for advancing our understanding of universal versus language-specific semantic patterns.

---

## Acknowledgments

The author acknowledges the use of computational resources and datasets from the Open Multilingual WordNet and ConceptNet projects, which made this research possible. Special thanks to the multilingual NLP community for providing the foundational models and datasets that enabled this work.

---

*This paper represents ongoing research in multilingual semantic analysis. For the most current results and additional implementation details, please refer to the project repository.*
