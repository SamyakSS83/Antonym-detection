import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.data import DataLoader as GeoDataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dualenc_frombert import DualEncoderGraphTransformer, WordPairGraphDataset, load_data

# Create assets directory if it doesn't exist
os.makedirs("diagnostic_results", exist_ok=True)

# -----------------------
# Device Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load Fine-Tuned BERT for Embeddings
# -----------------------
finetuned_bert_path = "./output/kaggle/working/assets/best_bert_model.pt"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ft_model = AutoModelForSequenceClassification.from_pretrained(model_name)
ft_model.load_state_dict(torch.load(finetuned_bert_path, map_location=device))
finetuned_bert = ft_model.bert.to(device)
finetuned_bert.eval()

# -----------------------
# Model Diagnostics Class
# -----------------------
class DualEncoderDiagnostics:
    def __init__(self, model_path, tokenizer, bert_encoder, device):
        self.tokenizer = tokenizer
        self.bert_encoder = bert_encoder
        self.device = device
        
        # Load the model
        self.model = DualEncoderGraphTransformer(in_channels=768, hidden_channels=256, out_channels=2).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def analyze_model_weights(self):
        """Analyze the model weights and biases."""
        print("\n## Model Weight Analysis")
        
        # Analyze synonym and antonym projection layers
        print("\n**Projection Layer Analysis**")
        
        # Synonym projection weights
        syn_w = self.model.syn_proj[0].weight
        print(f"Synonym projection weights - Shape: {syn_w.shape}, Mean: {syn_w.mean().item():.4f}, Std: {syn_w.std().item():.4f}")
        
        # Antonym projection weights
        ant_w = self.model.ant_proj[0].weight
        print(f"Antonym projection weights - Shape: {ant_w.shape}, Mean: {ant_w.mean().item():.4f}, Std: {ant_w.std().item():.4f}")
        
        # Fusion layer weights
        fusion_w = self.model.fusion.weight
        print(f"Fusion layer weights - Shape: {fusion_w.shape}, Mean: {fusion_w.mean().item():.4f}, Std: {fusion_w.std().item():.4f}")
        
        # First transformer layer
        conv1_key_w = self.model.conv1.lin_key.weight if hasattr(self.model.conv1, 'lin_key') else self.model.conv1.weight
        print(f"First transformer layer weights - Shape: {conv1_key_w.shape}, Mean: {conv1_key_w.mean().item():.4f}, Std: {conv1_key_w.std().item():.4f}")
        
        # Plot weight distributions
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(syn_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.title("Synonym Projection Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        plt.subplot(2, 2, 2)
        plt.hist(ant_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.title("Antonym Projection Weights")
        plt.xlabel("Weight Value")
        
        plt.subplot(2, 2, 3)
        plt.hist(fusion_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.title("Fusion Layer Weights")
        plt.xlabel("Weight Value")
        
        plt.subplot(2, 2, 4)
        plt.hist(conv1_key_w.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.title("First Transformer Layer Weights")
        plt.xlabel("Weight Value")
        
        plt.tight_layout()
        plt.savefig("diagnostic_results/weight_distributions.png")
        plt.close()
        
        print("Weight distribution plots saved to diagnostic_results/weight_distributions.png")
        
        # Analyze weight differences between synonym and antonym projections
        weight_diff = syn_w - ant_w
        print(f"\n**Projection Weight Differences**")
        print(f"Syn-Ant weight difference - Mean: {weight_diff.mean().item():.4f}, Std: {weight_diff.std().item():.4f}")
        
        plt.figure(figsize=(10, 5))
        plt.hist(weight_diff.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.title("Synonym-Antonym Projection Weight Differences")
        plt.xlabel("Weight Difference")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("diagnostic_results/projection_weight_diff.png")
        plt.close()
        
        print("Projection weight difference plot saved to diagnostic_results/projection_weight_diff.png")
        
    def analyze_dual_spaces(self, test_loader):
        """Analyze the dual projection spaces."""
        print("\n## Dual Projection Space Analysis")
        
        syn_similarities = {0: [], 1: []}  # {label: [similarities]}
        ant_similarities = {0: [], 1: []}
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Analyzing dual spaces"):
                batch = batch.to(device)
                
                # Forward pass to get projection outputs
                _ = self.model(batch)
                x_syn = self.model.x_syn
                x_ant = self.model.x_ant
                
                # Process each graph in the batch
                batch_np = batch.batch.cpu().numpy()
                unique_graphs = np.unique(batch_np)
                
                for g in unique_graphs:
                    idx = (batch.batch == g).nonzero(as_tuple=False).view(-1)
                    if idx.shape[0] != 2:
                        continue
                    
                    # Get graph label
                    graph_idx = g.item() if hasattr(g, 'item') else g
                    label = batch.y[graph_idx].item()
                    
                    # Compute similarities
                    sim_syn = torch.tanh(torch.dot(x_syn[idx[0]], x_syn[idx[1]])).item()
                    sim_ant = torch.tanh(torch.dot(x_ant[idx[0]], x_ant[idx[1]])).item()
                    
                    syn_similarities[label].append(sim_syn)
                    ant_similarities[label].append(sim_ant)
        
        # Analyze similarities
        print("\n**Dual Space Similarity Analysis**")
        print(f"Synonym space - Non-antonyms: {np.mean(syn_similarities[0]):.4f}, Antonyms: {np.mean(syn_similarities[1]):.4f}")
        print(f"Antonym space - Non-antonyms: {np.mean(ant_similarities[0]):.4f}, Antonyms: {np.mean(ant_similarities[1]):.4f}")
        
        # Plot similarity distributions
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.boxplot([syn_similarities[0], syn_similarities[1]], labels=['Non-Antonyms', 'Antonyms'])
        plt.title("Similarities in Synonym Space")
        plt.ylabel("Similarity (tanh)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([ant_similarities[0], ant_similarities[1]], labels=['Non-Antonyms', 'Antonyms'])
        plt.title("Similarities in Antonym Space")
        plt.ylabel("Similarity (tanh)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("diagnostic_results/dual_space_similarities.png")
        plt.close()
        
        print("Dual space similarity plots saved to diagnostic_results/dual_space_similarities.png")
        
        # Check if the model learned the expected patterns
        syn_diff = np.mean(syn_similarities[0]) - np.mean(syn_similarities[1])
        ant_diff = np.mean(ant_similarities[1]) - np.mean(ant_similarities[0])
        
        print("\n**Dual Encoder Learning Assessment**")
        print(f"Synonym space differentiation: {syn_diff:.4f} (higher is better)")
        print(f"Antonym space differentiation: {ant_diff:.4f} (higher is better)")
        
        if syn_diff > 0:
            print("✓ Model correctly learned to represent non-antonyms with higher similarity in synonym space")
        else:
            print("✗ Model failed to learn expected pattern in synonym space")
            
        if ant_diff > 0:
            print("✓ Model correctly learned to represent antonyms with higher similarity in antonym space")
        else:
            print("✗ Model failed to learn expected pattern in antonym space")
        
    def analyze_activations(self, test_loader, sample_batches=5):
        """Analyze activations through the network."""
        print("\n## Activation Analysis")
        
        activations = {
            "syn_proj_linear": [],
            "syn_proj_relu": [],
            "ant_proj_linear": [],
            "ant_proj_relu": [],
            "fusion": [],
            "conv1": [],
            "conv1_relu": [],
            "conv2": [],
            "pooled": [],
            "lin1": [],
            "lin1_relu": [],
            "output": []
        }
        
        self.model.eval()
        
        for i, batch in enumerate(test_loader):
            if i >= sample_batches:
                break
                
            batch = batch.to(device)
            
            with torch.no_grad():
                # Input nodes
                x, edge_index, batch_indices = batch.x, batch.edge_index, batch.batch
                
                # Projection branches
                x_syn_linear = self.model.syn_proj[0](x)
                x_syn_relu = self.model.syn_proj[1](x_syn_linear)
                
                x_ant_linear = self.model.ant_proj[0](x)
                x_ant_relu = self.model.ant_proj[1](x_ant_linear)
                
                # Fusion
                x_combined = torch.cat([x_syn_relu, x_ant_relu], dim=1)
                x_fused = self.model.fusion(x_combined)
                
                # Graph transformer layers
                x_conv1 = self.model.conv1(x_fused, edge_index)
                x_conv1_relu = nn.functional.relu(x_conv1)
                
                x_conv2 = self.model.conv2(x_conv1_relu, edge_index)
                
                # Pooling
                x_pooled = global_mean_pool(x_conv2, batch_indices)
                
                # Final layers
                x_lin1 = self.model.lin1(x_pooled)
                x_lin1_relu = nn.functional.relu(x_lin1)
                
                output = self.model.lin2(x_lin1_relu)
                
                # Store activations
                activations["syn_proj_linear"].extend(x_syn_linear.cpu().numpy().flatten())
                activations["syn_proj_relu"].extend(x_syn_relu.cpu().numpy().flatten())
                activations["ant_proj_linear"].extend(x_ant_linear.cpu().numpy().flatten())
                activations["ant_proj_relu"].extend(x_ant_relu.cpu().numpy().flatten())
                activations["fusion"].extend(x_fused.cpu().numpy().flatten())
                activations["conv1"].extend(x_conv1.cpu().numpy().flatten())
                activations["conv1_relu"].extend(x_conv1_relu.cpu().numpy().flatten())
                activations["conv2"].extend(x_conv2.cpu().numpy().flatten())
                activations["pooled"].extend(x_pooled.cpu().numpy().flatten())
                activations["lin1"].extend(x_lin1.cpu().numpy().flatten())
                activations["lin1_relu"].extend(x_lin1_relu.cpu().numpy().flatten())
                activations["output"].extend(output.cpu().numpy().flatten())
        
        # Print activation statistics
        print("\n**Activation Statistics**")
        for name, values in activations.items():
            print(f"{name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}, Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")
        
        # Plot activation distributions
        plt.figure(figsize=(15, 15))
        
        keys = list(activations.keys())
        for i, key in enumerate(keys):
            plt.subplot(4, 3, i+1)
            plt.hist(activations[key], bins=50, alpha=0.7)
            plt.title(key)
            plt.xlabel("Activation Value")
            if i % 3 == 0:
                plt.ylabel("Frequency")
                
        plt.tight_layout()
        plt.savefig("diagnostic_results/activation_distributions.png")
        plt.close()
        
        print("Activation distribution plots saved to diagnostic_results/activation_distributions.png")
        
    def analyze_predictions(self, test_loader):
        """Analyze prediction patterns and errors."""
        print("\n## Prediction Analysis")
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Analyzing predictions"):
                batch = batch.to(device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_logits.extend(logits.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["Not Antonym", "Antonym"])
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"Overall Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Antonym", "Antonym"], yticklabels=["Not Antonym", "Antonym"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("diagnostic_results/confusion_matrix.png")
        plt.close()
        
        # Analyze confidence of predictions
        all_logits = np.array(all_logits)
        all_confidences = softmax(all_logits, axis=1)
        
        # Get confidence of predicted class
        pred_confidences = np.array([all_confidences[i, p] for i, p in enumerate(all_preds)])
        
        # True positives, true negatives, false positives, false negatives
        tp = np.logical_and(all_preds == 1, all_labels == 1)
        tn = np.logical_and(all_preds == 0, all_labels == 0)
        fp = np.logical_and(all_preds == 1, all_labels == 0)
        fn = np.logical_and(all_preds == 0, all_labels == 1)
        
        # Plot confidence by prediction type
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(pred_confidences[tp], bins=20, alpha=0.7, label='True Positives')
        plt.hist(pred_confidences[tn], bins=20, alpha=0.7, label='True Negatives')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence of Correct Predictions')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(pred_confidences[fp], bins=20, alpha=0.7, label='False Positives')
        plt.hist(pred_confidences[fn], bins=20, alpha=0.7, label='False Negatives')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence of Incorrect Predictions')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("diagnostic_results/prediction_confidence.png")
        plt.close()
        
        print("Prediction analysis plots saved to diagnostic_results/prediction_confidence.png")
        
        # Report average confidence by prediction type
        print("\n**Prediction Confidence Analysis**")
        print(f"True Positives: {np.mean(pred_confidences[tp]):.4f} (std: {np.std(pred_confidences[tp]):.4f})")
        print(f"True Negatives: {np.mean(pred_confidences[tn]):.4f} (std: {np.std(pred_confidences[tn]):.4f})")
        print(f"False Positives: {np.mean(pred_confidences[fp]):.4f} (std: {np.std(pred_confidences[fp]):.4f})")
        print(f"False Negatives: {np.mean(pred_confidences[fn]):.4f} (std: {np.std(pred_confidences[fn]):.4f})")

def softmax(x, axis=None):
    """Compute softmax values for each row of x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
        
def main():
    """Main function for model diagnosis."""
    print("## BERT Dual Encoder Model Diagnosis")
    
    # Define paths
    data_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    model_path = "output/kaggle/working/assets/best_dual_encoder_graph_model.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.DataFrame()
    for wt in word_types:
        test_file = os.path.join(data_dir, f"{wt}.test")
        if os.path.exists(test_file):
            test_df = pd.concat([test_df, load_data(test_file)], ignore_index=True)
    
    if len(test_df) == 0:
        print("Error: No test data found.")
        return
    
    print(f"Loaded {len(test_df)} test samples")
    
    # Create dataset and dataloader
    test_dataset = WordPairGraphDataset(test_df)
    test_loader = GeoDataLoader(test_dataset, batch_size=32)
    
    # Initialize diagnostics
    diagnostics = DualEncoderDiagnostics(
        model_path=model_path,
        tokenizer=tokenizer,
        bert_encoder=finetuned_bert,
        device=device
    )
    
    # Run analyses
    diagnostics.analyze_model_weights()
    diagnostics.analyze_dual_spaces(test_loader)
    diagnostics.analyze_activations(test_loader)
    diagnostics.analyze_predictions(test_loader)
    
    # Also analyze per word type
    print("\n## Word Type Analysis")
    for wt in word_types:
        test_file = os.path.join(data_dir, f"{wt}.test")
        if os.path.exists(test_file):
            print(f"\nAnalyzing {wt}...")
            wt_df = load_data(test_file)
            wt_dataset = WordPairGraphDataset(wt_df)
            wt_loader = GeoDataLoader(wt_dataset, batch_size=32)
            
            # Analyze predictions for this word type
            all_preds = []
            all_labels = []
            
            diagnostics.model.eval()
            with torch.no_grad():
                for batch in wt_loader:
                    batch = batch.to(device)
                    logits = diagnostics.model(batch)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = batch.y.cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            acc = accuracy_score(all_labels, all_preds)
            report = classification_report(all_labels, all_preds, target_names=["Not Antonym", "Antonym"])
            cm = confusion_matrix(all_labels, all_preds)
            
            print(f"{wt} Accuracy: {acc:.4f}")
            print(f"{wt} Classification Report:")
            print(report)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Antonym", "Antonym"], yticklabels=["Not Antonym", "Antonym"])
            plt.title(f"{wt} Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(f"diagnostic_results/{wt}_confusion_matrix.png")
            plt.close()
    
    print("\nDiagnosis complete! Results saved to the diagnostic_results directory.")

if __name__ == "__main__":
    main()