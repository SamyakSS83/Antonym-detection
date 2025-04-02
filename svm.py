import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

def evaluate_model(model, X_test, y_test, dataset_name=""):
    """Evaluate model and print metrics."""
    # Reshape input if needed
    if len(X_test.shape) > 2:
        X_test_reshaped = X_test.reshape(len(X_test), -1)
    else:
        X_test_reshaped = X_test
    
    # Get predictions
    y_pred = model.predict(X_test_reshaped)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
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
    plt.savefig(f'assets/confusion_matrix_{dataset_name.replace(" ", "_")}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def main():
    # Define paths
    dataset_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    
    # Initialize the model
    print("Loading Nomic embedding model...")
    model_st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # 1. Process each word type separately
    all_results = {}
    
    for word_type in word_types:
        print(f"\n=== Processing {word_type} ===")
        
        # Load training and validation data
        train_file = os.path.join(dataset_dir, f"{word_type}.train")
        val_file = os.path.join(dataset_dir, f"{word_type}.val")
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        
        # Load data
        word1_train, word2_train, y_train = load_data(train_file)
        word1_val, word2_val, y_val = load_data(val_file)
        word1_test, word2_test, y_test = load_data(test_file)
        
        # Combine train and val data
        word1_combined = word1_train + word1_val
        word2_combined = word2_train + word2_val
        y_combined = y_train + y_val
        
        print(f"Combined training data: {len(word1_combined)} samples")
        print(f"Test data: {len(word1_test)} samples")
        
        # Generate embeddings
        X_combined = embed_word_pairs(word1_combined, word2_combined, model_st)
        X_test = embed_word_pairs(word1_test, word2_test, model_st)
        
        # Reshape for SVM
        X_combined_reshaped = X_combined.reshape(len(X_combined), -1)
        X_test_reshaped = X_test.reshape(len(X_test), -1)
        
        # Train SVM model
        print("Training Polynomial SVM...")
        svm_clf = SVC(kernel='poly', degree=4, verbose=True)
        svm_clf.fit(X_combined_reshaped, y_combined)
        
        # Evaluate model
        results = evaluate_model(svm_clf, X_test_reshaped, y_test, dataset_name=word_type)
        all_results[word_type] = results
        
        # Save the model
        joblib.dump(svm_clf, f"svm_model_{word_type}.pkl")
        print(f"Model saved as svm_model_{word_type}.pkl")
    
    # Calculate and print overall metrics for type-specific models
    all_accuracies = [results['accuracy'] for results in all_results.values()]
    avg_accuracy = np.mean(all_accuracies)
    
    print("\n=== Overall Results for Type-Specific Models ===")
    print(f"Average accuracy across all word types: {avg_accuracy:.4f}")
    
    for word_type, results in all_results.items():
        print(f"{word_type} accuracy: {results['accuracy']:.4f}")
    
    # 2. Train a combined model on ALL data (train+val+test from all types)
    print("\n\n=== Training Full Dataset Model ===")
    
    # Collect all data across all word types and splits
    all_word1, all_word2, all_labels = [], [], []
    
    for word_type in word_types:
        for split in ["train", "val", "test"]:
            file_path = os.path.join(dataset_dir, f"{word_type}.{split}")
            w1, w2, labels = load_data(file_path)
            all_word1.extend(w1)
            all_word2.extend(w2)
            all_labels.extend(labels)
    
    print(f"Total combined dataset size: {len(all_word1)} samples")
    
    # Generate embeddings for the full dataset
    X_full = embed_word_pairs(all_word1, all_word2, model_st)
    X_full_reshaped = X_full.reshape(len(X_full), -1)
    
    # Train SVM on the full dataset
    print("Training Full Dataset Polynomial SVM...")
    full_svm_clf = SVC(kernel='poly', degree=4, verbose=True)
    full_svm_clf.fit(X_full_reshaped, all_labels)
    
    # Save the full model
    joblib.dump(full_svm_clf, "svm_model_full_dataset.pkl")
    print("Full dataset model saved as svm_model_full_dataset.pkl")
    
    # Evaluate the full model on each word type's test set for comparison
    print("\n=== Evaluating Full Dataset Model on Test Sets ===")
    full_model_results = {}
    
    for word_type in word_types:
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        word1_test, word2_test, y_test = load_data(test_file)
        
        X_test = embed_word_pairs(word1_test, word2_test, model_st)
        X_test_reshaped = X_test.reshape(len(X_test), -1)
        
        results = evaluate_model(full_svm_clf, X_test_reshaped, y_test, 
                               dataset_name=f"Full Model on {word_type}")
        full_model_results[word_type] = results
    
    # Calculate and print overall metrics for the full model
    full_model_accuracies = [results['accuracy'] for results in full_model_results.values()]
    full_model_avg_accuracy = np.mean(full_model_accuracies)
    
    print("\n=== Overall Results for Full Dataset Model ===")
    print(f"Average accuracy across all word types: {full_model_avg_accuracy:.4f}")
    
    for word_type, results in full_model_results.items():
        print(f"{word_type} accuracy: {results['accuracy']:.4f}")
    
    # Compare type-specific models vs full model
    print("\n=== Model Comparison: Type-Specific vs Full Dataset Model ===")
    comparison_data = []
    
    for word_type in word_types:
        type_specific_acc = all_results[word_type]['accuracy']
        full_model_acc = full_model_results[word_type]['accuracy']
        diff = full_model_acc - type_specific_acc
        comparison_data.append({
            'Word Type': word_type,
            'Type-Specific Model': type_specific_acc,
            'Full Dataset Model': full_model_acc,
            'Difference': diff
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to model_comparison.csv")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(word_types))
    
    plt.bar(index, [all_results[wt]['accuracy'] for wt in word_types], 
            bar_width, label='Type-Specific Model')
    plt.bar(index + bar_width, [full_model_results[wt]['accuracy'] for wt in word_types], 
            bar_width, label='Full Dataset Model')
    
    plt.xlabel('Word Type')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width / 2, word_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/model_comparison.png')
    plt.close()
    
    print("Model comparison plot saved as model_comparison.png")

if __name__ == "__main__":
    main()