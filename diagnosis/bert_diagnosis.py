import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load the datasets
def load_data(file_path):
    """Load data from a file into a DataFrame."""
    word1_list, word2_list, labels = [], [], []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    
    return pd.DataFrame({
        'word1': word1_list,
        'word2': word2_list,
        'label': labels
    })


# Create a custom dataset
class WordPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.word_pairs = []
        
        for i in range(len(dataframe)):
            self.word_pairs.append((
                str(dataframe.iloc[i]['word1']),
                str(dataframe.iloc[i]['word2'])
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word1 = str(self.data.iloc[idx]['word1'])
        word2 = str(self.data.iloc[idx]['word2'])
        label = int(self.data.iloc[idx]['label'])

        if word1 == 'nan' or word2 == 'nan':
            word1 = "unknown" if word1 == 'nan' else word1
            word2 = "unknown" if word2 == 'nan' else word2

        encoding = self.tokenizer.encode_plus(
            word1,
            word2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'word1': word1,
            'word2': word2
        }


class ModelDiagnostics:
    def __init__(self, model_path, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
        # Load the classification model
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Also load the base BERT model for extracting embeddings
        self.base_model = BertModel.from_pretrained('bert-base-uncased')
        self.base_model.to(device)
        self.base_model.eval()
        
        # Create output directory
        os.makedirs("model_diagnostics", exist_ok=True)
        
    def analyze_performance(self, test_loader, word_types=None):
        """Analyze model performance on test data"""
        print("\n## Performance Analysis")
        
        all_preds = []
        all_true = []
        all_probs = []
        all_word_pairs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating model"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                for i in range(len(batch['word1'])):
                    all_word_pairs.append((batch['word1'][i], batch['word2'][i]))
        
        # Overall performance
        accuracy = accuracy_score(all_true, all_preds)
        report = classification_report(all_true, all_preds, target_names=['Not Antonym', 'Antonym'])
        conf_matrix = confusion_matrix(all_true, all_preds)
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Antonym', 'Antonym'],
                   yticklabels=['Not Antonym', 'Antonym'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('model_diagnostics/confusion_matrix.png')
        
        # Analyze confidence distribution
        plt.figure(figsize=(10, 6))
        correct_probs = [all_probs[i][all_true[i]] for i in range(len(all_true))]
        plt.hist(correct_probs, bins=20, alpha=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Distribution of Model Confidence for Correct Class')
        plt.savefig('model_diagnostics/confidence_distribution.png')
        
        # Find most confident correct and incorrect predictions
        results_df = pd.DataFrame({
            'word1': [pair[0] for pair in all_word_pairs],
            'word2': [pair[1] for pair in all_word_pairs],
            'true_label': all_true,
            'pred_label': all_preds,
            'confidence': [all_probs[i][all_preds[i]] for i in range(len(all_preds))]
        })
        
        results_df['correct'] = results_df['true_label'] == results_df['pred_label']
        
        # Save all results
        results_df.to_csv('model_diagnostics/prediction_results.csv', index=False)
        
        # Most confident correct predictions
        top_correct = results_df[results_df['correct']].sort_values('confidence', ascending=False).head(20)
        print("\nMost confident correct predictions:")
        print(top_correct[['word1', 'word2', 'true_label', 'confidence']])
        
        # Most confident incorrect predictions
        top_incorrect = results_df[~results_df['correct']].sort_values('confidence', ascending=False).head(20)
        print("\nMost confident incorrect predictions:")
        print(top_incorrect[['word1', 'word2', 'true_label', 'pred_label', 'confidence']])
        
        return results_df
    
    def extract_embeddings(self, dataloader, layer_indices=[-1, -2, -4, -8]):
        """Extract embeddings from different layers of the model"""
        print("\n## Extracting Embeddings from Different Layers")
        
        # Dictionary to store embeddings from different layers
        all_embeddings = {f"layer_{i}": [] for i in layer_indices}
        all_cls_embeddings = {f"layer_{i}": [] for i in layer_indices}
        all_labels = []
        all_word_pairs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get all hidden states
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states
                
                # Extract embeddings from specified layers
                for layer_idx in layer_indices:
                    layer_output = hidden_states[layer_idx]
                    
                    # Get CLS token embeddings (first token)
                    cls_embeddings = layer_output[:, 0, :].cpu().numpy()
                    all_cls_embeddings[f"layer_{layer_idx}"].extend(cls_embeddings)
                    
                    # Get mean of all token embeddings (excluding padding)
                    # Create a mask to exclude padding tokens - MOVE TO SAME DEVICE AS LAYER_OUTPUT
                    mask = batch['attention_mask'].unsqueeze(-1).expand(layer_output.size()).float().to(self.device)
                    # Sum up all token embeddings and divide by the number of tokens
                    sum_embeddings = torch.sum(layer_output * mask, 1)
                    token_counts = torch.clamp(torch.sum(mask, 1), min=1e-9)
                    mean_embeddings = (sum_embeddings / token_counts).cpu().numpy()
                    all_embeddings[f"layer_{layer_idx}"].extend(mean_embeddings)
                
                all_labels.extend(labels.numpy())
                for i in range(len(batch['word1'])):
                    all_word_pairs.append((batch['word1'][i], batch['word2'][i]))
        
        return all_embeddings, all_cls_embeddings, all_labels, all_word_pairs
    
    def visualize_embeddings(self, embeddings, labels, word_pairs, method='tsne', layer='layer_-1'):
        """Visualize embeddings using dimensionality reduction"""
        print(f"\n## Visualizing Embeddings using {method.upper()} for {layer}")
        
        # Convert embeddings to numpy array
        X = np.array(embeddings[layer])
        y = np.array(labels)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(X)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(X)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(X)
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': y,
            'word1': [pair[0] for pair in word_pairs],
            'word2': [pair[1] for pair in word_pairs]
        })
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='label', 
                        palette={0: 'blue', 1: 'red'},
                        alpha=0.7, s=50)
        
        plt.title(f'Embedding Visualization using {method.upper()} for {layer}')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.legend(title='Label', labels=['Not Antonym', 'Antonym'])
        
        # Add annotations for some points
        np.random.seed(42)
        sample_indices = np.random.choice(len(df), size=20, replace=False)
        for idx in sample_indices:
            plt.annotate(f"{df.iloc[idx]['word1']}-{df.iloc[idx]['word2']}",
                        (df.iloc[idx]['x'], df.iloc[idx]['y']),
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'model_diagnostics/{method}_{layer}_visualization.png', dpi=300)
    
    def analyze_attention_patterns(self, word_pairs, labels, n_samples=10):
        """Analyze attention patterns for specific word pairs"""
        print("\n## Analyzing Attention Patterns")
        
        # Create a directory for attention visualizations
        os.makedirs("model_diagnostics/attention", exist_ok=True)
        
        # Select a subset of samples to analyze
        np.random.seed(42)
        if len(word_pairs) > n_samples:
            indices = np.random.choice(len(word_pairs), size=n_samples, replace=False)
            selected_pairs = [word_pairs[i] for i in indices]
            selected_labels = [labels[i] for i in indices]
        else:
            selected_pairs = word_pairs
            selected_labels = labels
        
        attention_scores = []
        
        for i, (pair, label) in enumerate(zip(selected_pairs, selected_labels)):
            word1, word2 = pair
            
            # Tokenize the input
            inputs = self.tokenizer(word1, word2, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get token strings for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Forward pass with output_attentions=True
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            
            # Get attention weights (12 layers, each with attention heads)
            attentions = outputs.attentions
            
            # Average attention across heads for each layer
            avg_attentions = [layer_attn.mean(dim=1).cpu().numpy() for layer_attn in attentions]
            
            # Store for later analysis
            attention_scores.append({
                'word1': word1,
                'word2': word2,
                'label': label,
                'tokens': tokens,
                'attentions': avg_attentions
            })
            
            # Visualize attention for a few samples
            if i < 5:
                # Visualize attention from the last layer
                last_layer_attn = avg_attentions[-1][0]  # Shape: [seq_len, seq_len]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(last_layer_attn, cmap='viridis', 
                           xticklabels=tokens, yticklabels=tokens)
                plt.title(f"Attention Pattern: '{word1}' - '{word2}' (Label: {label})")
                plt.tight_layout()
                plt.savefig(f'model_diagnostics/attention/attn_pair_{i}.png')
        
        # Analyze attention to special tokens
        print("\nAnalyzing attention to special tokens...")
        cls_attention = defaultdict(list)
        sep_attention = defaultdict(list)
        
        for sample in attention_scores:
            tokens = sample['tokens']
            
            # Find positions of special tokens
            cls_pos = tokens.index('[CLS]')
            sep_positions = [i for i, t in enumerate(tokens) if t == '[SEP]']
            
            # For each layer, measure attention to special tokens
            for layer_idx, layer_attn in enumerate(sample['attentions']):
                # How much other tokens attend to [CLS]
                cls_attention[layer_idx].append(layer_attn[0, :, cls_pos].mean())
                
                # How much other tokens attend to [SEP] tokens
                for sep_pos in sep_positions:
                    sep_attention[layer_idx].append(layer_attn[0, :, sep_pos].mean())
        
        # Plot attention to special tokens across layers
        plt.figure(figsize=(10, 6))
        layers = list(cls_attention.keys())
        plt.plot(layers, [np.mean(cls_attention[l]) for l in layers], 'o-', label='[CLS] token')
        plt.plot(layers, [np.mean(sep_attention[l]) for l in layers], 's-', label='[SEP] token')
        plt.xlabel('Layer')
        plt.ylabel('Average Attention')
        plt.title('Attention to Special Tokens Across Layers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('model_diagnostics/attention_to_special_tokens.png')
        
        return attention_scores
    
    def analyze_feature_importance(self, word_pairs, labels, n_samples=10):
        """Analyze feature importance using integrated gradients"""
        print("\n## Analyzing Feature Importance")
        
        # Create a directory for feature importance visualizations
        os.makedirs("model_diagnostics/feature_importance", exist_ok=True)
        
        # Select a subset of samples to analyze
        np.random.seed(42)
        if len(word_pairs) > n_samples:
            indices = np.random.choice(len(word_pairs), size=n_samples, replace=False)
            selected_pairs = [word_pairs[i] for i in indices]
            selected_labels = [labels[i] for i in indices]
        else:
            selected_pairs = word_pairs
            selected_labels = labels
        
        # Create a wrapper function that returns only the logits
        def model_wrapper(input_ids, attention_mask=None):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return output.logits
        
        # Initialize LayerIntegratedGradients with the wrapper
        lig = LayerIntegratedGradients(
            forward_func=model_wrapper, 
            layer=self.model.bert.embeddings.word_embeddings
        )
        
        for i, (pair, label) in enumerate(zip(selected_pairs, selected_labels)):
            word1, word2 = pair
            
            # Tokenize the input
            inputs = self.tokenizer(word1, word2, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get token strings for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Forward pass to get prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_class = torch.argmax(outputs.logits, dim=1).item()
            
            # Create baseline (all [PAD] tokens)
            baseline_input_ids = torch.zeros_like(input_ids).to(self.device)
            
            # Calculate attributions
            attributions, delta = lig.attribute(
                inputs=input_ids,
                baselines=baseline_input_ids,
                target=pred_class,
                additional_forward_args=(attention_mask,),
                return_convergence_delta=True
            )
            
            # Sum attributions across embedding dimension
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions.detach().cpu().numpy()
            
            # Normalize attributions for visualization
            attributions = attributions / np.linalg.norm(attributions)
            
            # Create a DataFrame for visualization
            attr_df = pd.DataFrame({
                'token': tokens,
                'attribution': attributions
            })
            
            # Plot attributions
            plt.figure(figsize=(12, 6))
            colors = ['red' if a < 0 else 'green' for a in attributions]
            plt.bar(range(len(tokens)), attributions, color=colors)
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.xlabel('Tokens')
            plt.ylabel('Attribution Score')
            plt.title(f"Token Attributions for '{word1}'-'{word2}' (Label: {label}, Pred: {pred_class})")
            plt.tight_layout()
            plt.savefig(f'model_diagnostics/feature_importance/attr_pair_{i}.png')
            
            # Save top positive and negative attributions
            attr_df = attr_df.sort_values('attribution', ascending=False)
            print(f"\nPair: '{word1}'-'{word2}' (Label: {label}, Pred: {pred_class})")
            print("Top positive attributions:")
            print(attr_df.head(5))
            print("Top negative attributions:")
            print(attr_df.tail(5))
        
        return True

    def analyze_neuron_activations(self, dataloader, n_samples=100):
        """Analyze individual neuron activations to find interpretable patterns"""
        print("\n## Analyzing Neuron Activations")
        
        # Create directory for neuron activation visualizations
        os.makedirs("model_diagnostics/neuron_activations", exist_ok=True)
        
        # Collect samples and their activations
        all_activations = []
        all_word_pairs = []
        all_labels = []
        
        # Get a subset of samples
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting neuron activations"):
                if sample_count >= n_samples:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get all hidden states
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get the last hidden state (before classification)
                last_hidden_state = outputs.hidden_states[-1]
                
                # Get CLS token representation (used for classification)
                cls_activations = last_hidden_state[:, 0, :].cpu().numpy()
                
                # Add to our collections
                all_activations.extend(cls_activations)
                all_labels.extend(labels.cpu().numpy())
                
                for i in range(len(batch['input_ids'])):
                    if 'word1' in batch and 'word2' in batch:
                        all_word_pairs.append((batch['word1'][i], batch['word2'][i]))
                    else:
                        all_word_pairs.append(("unknown", "unknown"))
                
                sample_count += len(batch['input_ids'])
                if sample_count >= n_samples:
                    break
        
        # Convert to numpy arrays
        activations = np.array(all_activations)
        labels = np.array(all_labels)
        
        # Find neurons with highest activation variance
        activation_variance = np.var(activations, axis=0)
        top_neurons = np.argsort(-activation_variance)[:20]  # Top 20 neurons by variance
        
        # Analyze top neurons
        print("\nTop neurons by activation variance:")
        for i, neuron_idx in enumerate(top_neurons):
            # Get activations for this neuron
            neuron_activations = activations[:, neuron_idx]
            
            # Find samples with highest and lowest activations
            highest_indices = np.argsort(-neuron_activations)[:5]
            lowest_indices = np.argsort(neuron_activations)[:5]
            
            print(f"\nNeuron {neuron_idx}:")
            print("  Highest activating samples:")
            for idx in highest_indices:
                if idx < len(all_word_pairs):
                    print(f"    {all_word_pairs[idx][0]}-{all_word_pairs[idx][1]} (Label: {all_labels[idx]}, Activation: {neuron_activations[idx]:.4f})")
            
            print("  Lowest activating samples:")
            for idx in lowest_indices:
                if idx < len(all_word_pairs):
                    print(f"    {all_word_pairs[idx][0]}-{all_word_pairs[idx][1]} (Label: {all_labels[idx]}, Activation: {neuron_activations[idx]:.4f})")
            
            # Plot activation distribution by class
            plt.figure(figsize=(10, 6))
            for label_value in np.unique(labels):
                label_activations = neuron_activations[labels == label_value]
                plt.hist(label_activations, alpha=0.5, bins=20, 
                        label=f"Class {label_value}")
            
            plt.title(f"Neuron {neuron_idx} Activation Distribution")
            plt.xlabel("Activation Value")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(f"model_diagnostics/neuron_activations/neuron_{neuron_idx}_dist.png")
        
        # Find neurons that best separate the classes
        class_separability = []
        for i in range(activations.shape[1]):
            pos_activations = activations[labels == 1, i]
            neg_activations = activations[labels == 0, i]
            
            if len(pos_activations) > 0 and len(neg_activations) > 0:
                # Calculate separation using difference in means normalized by pooled standard deviation
                mean_diff = abs(np.mean(pos_activations) - np.mean(neg_activations))
                pooled_std = np.sqrt((np.var(pos_activations) + np.var(neg_activations)) / 2)
                
                if pooled_std > 0:
                    separability = mean_diff / pooled_std
                else:
                    separability = 0
            else:
                separability = 0
                
            class_separability.append(separability)
        
        # Get top neurons by class separability
        top_separating_neurons = np.argsort(-np.array(class_separability))[:10]
        
        print("\nTop neurons by class separability:")
        for i, neuron_idx in enumerate(top_separating_neurons):
            print(f"Neuron {neuron_idx}: Separability score = {class_separability[neuron_idx]:.4f}")
        
        return top_neurons, top_separating_neurons

    def run_full_diagnostics(self, test_loader):
        """Run all diagnostic analyses"""
        print("## Running Full Model Diagnostics")
        
        # 1. Performance analysis
        results_df = self.analyze_performance(test_loader)
        
        # 2. Extract embeddings from different layers
        word_pairs = [(row['word1'], row['word2']) for _, row in results_df.iterrows()]
        labels = results_df['true_label'].tolist()
        
        # Extract a subset for detailed analysis
        np.random.seed(42)
        if len(word_pairs) > 500:
            indices = np.random.choice(len(word_pairs), size=500, replace=False)
            sample_pairs = [word_pairs[i] for i in indices]
            sample_labels = [labels[i] for i in indices]
            sample_df = results_df.iloc[indices]
        else:
            sample_pairs = word_pairs
            sample_labels = labels
            sample_df = results_df
        
        # Create a dataset and dataloader for the samples
        tokenizer = self.tokenizer
        
        class SampleDataset(Dataset):
            def __init__(self, pairs, labels, tokenizer, max_length=128):
                self.pairs = pairs
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                word1, word2 = self.pairs[idx]
                label = self.labels[idx]
                
                encoding = self.tokenizer.encode_plus(
                    word1,
                    word2,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long),
                    'word1': word1,
                    'word2': word2
                }
        
        sample_dataset = SampleDataset(sample_pairs, sample_labels, tokenizer)
        sample_loader = DataLoader(sample_dataset, batch_size=16)
        
        # 3. Extract embeddings
        embeddings, cls_embeddings, embed_labels, embed_pairs = self.extract_embeddings(sample_loader)
        
        # 4. Visualize embeddings using different methods
        for layer in embeddings.keys():
            self.visualize_embeddings(cls_embeddings, embed_labels, embed_pairs, method='tsne', layer=layer)
            self.visualize_embeddings(cls_embeddings, embed_labels, embed_pairs, method='pca', layer=layer)
        
        # 5. Analyze attention patterns
        attention_scores = self.analyze_attention_patterns(sample_pairs[:20], sample_labels[:20])
        
        # 6. Analyze feature importance
        self.analyze_feature_importance(sample_pairs[:20], sample_labels[:20])
        
        # 7. Analyze neuron activations
        top_neurons, top_separating_neurons = self.analyze_neuron_activations(sample_loader)
        
        print("\n## Diagnostics Complete")
        print("All results have been saved to the 'model_diagnostics' directory")
        
        return {
            'results': results_df,
            'embeddings': embeddings,
            'cls_embeddings': cls_embeddings,
            'attention_scores': attention_scores,
            'top_neurons': top_neurons,
            'top_separating_neurons': top_separating_neurons
        }


def main():
    # Path to data directory
    dataset_dir = "/kaggle/input/dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    
    # Load test data
    test_data = pd.DataFrame()
    for word_type in word_types:
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        if os.path.exists(test_file):
            test_data = pd.concat([test_data, load_data(test_file)])
    
    test_data = test_data.dropna()
    print(f"Test data: {len(test_data)} samples")
    
    # Initialize tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create test dataset and dataloader
    test_dataset = WordPairDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize the diagnostics tool
    model_path = "/kaggle/working/assets/best_bert_model.pt"
    diagnostics = ModelDiagnostics(model_path, tokenizer, device)
    
    # Run diagnostics
    results = diagnostics.run_full_diagnostics(test_loader)
    
    print("Diagnostics completed successfully!")

if __name__ == "__main__":
    main()
