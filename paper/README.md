# Bhav-Net: Multilingual Semantic Analysis Paper

This folder contains the research paper for Bhav-Net, a novel multilingual dual-encoder architecture for cross-lingual antonym-synonym detection.

## Paper Title
**Bhav-Net: Multilingual Semantic Analysis for Cross-Lingual Antonym-Synonym Detection Using Dual-Encoder Graph Transformers**

## Author
Samyak S. Sanghvi  
Department of Computer Science and Engineering  
Indian Institute of Technology Delhi  
Email: cs1230807@iitd.ac.in

## Abstract
This paper presents Bhav-Net, a novel multilingual dual-encoder architecture for antonym-synonym detection that effectively captures semantic relationships across seven languages. Building upon traditional dual encoder frameworks, Bhav-Net integrates language-specific fine-tuned BERT models with graph transformer networks to model word relationships as dual projections in semantic space.

## Key Contributions

1. **Dual-Space Projection Mechanism**: Explicitly models synonym and antonym relationships through separate projection branches
2. **Graph-Based Representation Framework**: Captures relational dependencies through transformer convolutions on two-node word pair graphs
3. **Multilingual Training Strategy**: Leverages language-specific BERT models for optimal cross-lingual performance
4. **Comprehensive Evaluation**: Demonstrates effectiveness across German, French, Spanish, Italian, Portuguese, Dutch, and Russian

## Files

- `bhav_net_paper.tex` - Main LaTeX source file for the paper
- `references.bib` - Bibliography file with all citations
- `README.md` - This file

## Architecture Overview

Bhav-Net employs a four-stage architecture:

1. **Language-Specific BERT Encoding**: Uses specialized BERT models for each language (e.g., CamemBERT for French, dbmdz/bert-base-german-cased for German)

2. **Dual Projection Branches**: 
   - Synonym projection: Projects embeddings into synonym space
   - Antonym projection: Projects embeddings into antonym space

3. **Graph Transformer Convolutions**: Models word pairs as two-node graphs with bidirectional edges, applying transformer convolutions to capture relational dependencies

4. **Classification with Margin Loss**: Combines standard classification loss with margin-based loss to enforce semantic space organization

## Mathematical Formulation

### Dual Projections
```
x_syn = Dropout(ReLU(W_syn·x + b_syn))
x_ant = Dropout(ReLU(W_ant·x + b_ant))
```

### Feature Fusion
```
x_fused = W_f[x_syn; x_ant] + b_f
```

### Margin Loss
```
L_margin = {
  max(0, m_syn - tanh(⟨x_syn¹, x_syn²⟩))  if y = 0 (synonym)
  max(0, tanh(⟨x_ant¹, x_ant²⟩) - m_ant)   if y = 1 (antonym)
}
```

## Training Strategy

**Two-Stage Training**:
1. **Stage 1**: Fine-tune language-specific BERT models for antonym classification (3-5 epochs)
2. **Stage 2**: Train dual encoder with frozen BERT, focusing on projection and graph learning (10-20 epochs)

## Data Sources

- **Open Multilingual WordNet (OMW)**: Professional linguistic data
- **ConceptNet**: Large-scale semantic knowledge graph
- **Real antonym pairs**: 2,263-6,095 pairs per language (100-300x improvement over synthetic data)

## Languages Supported

| Language   | BERT Model                              | Dataset Size |
|------------|----------------------------------------|--------------|
| German     | dbmdz/bert-base-german-cased          | 2,678 pairs  |
| French     | camembert-base                         | 6,095 pairs  |
| Spanish    | dccuchile/bert-base-spanish-wwm-cased | 2,263 pairs  |
| Italian    | dbmdz/bert-base-italian-cased         | 2,495 pairs  |
| Portuguese | neuralmind/bert-base-portuguese-cased | TBD          |
| Dutch      | GroNLP/bert-base-dutch-cased          | TBD          |
| Russian    | DeepPavlov/rubert-base-cased          | TBD          |

## Compilation Instructions

To compile the LaTeX paper:

```bash
# Standard LaTeX compilation
pdflatex bhav_net_paper.tex
bibtex bhav_net_paper
pdflatex bhav_net_paper.tex
pdflatex bhav_net_paper.tex

# Or using latexmk for automatic compilation
latexmk -pdf bhav_net_paper.tex
```

## Key Features of Bhav-Net

### Why it Works Across Languages

1. **Universal Semantic Principles**: The dual-space architecture models universal patterns of opposition and similarity that transcend linguistic boundaries

2. **Language-Specific Optimization**: Uses native BERT models that understand language-specific morphology, syntax, and semantics

3. **Graph-Based Relational Reasoning**: The transformer convolution architecture captures relational patterns that are consistent across languages

4. **Margin-Based Semantic Organization**: Forces clear separation between antonym and synonym relationships in learned representations

### Architectural Insights

The architecture closely symbolizes human understanding of word relationships:

- **Dual Spaces**: Mirrors how humans maintain separate mental models for similarity vs. opposition
- **Graph Structure**: Reflects the relational nature of semantic understanding
- **Attention Mechanisms**: Emulates selective focus on relevant semantic features
- **Margin Enforcement**: Creates clear decision boundaries similar to human categorization

## Research Impact

This work advances multilingual NLP by:
- Providing a robust framework for cross-lingual semantic relationship detection
- Demonstrating the effectiveness of explicit dual-space modeling
- Establishing new benchmarks for multilingual antonym detection
- Offering insights into universal vs. language-specific semantic patterns

## Future Work

- Extension to additional languages and language families
- Investigation of few-shot learning for low-resource languages
- Application to broader semantic relationship detection tasks
- Analysis of transfer learning capabilities across related languages
