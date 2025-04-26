#!/usr/bin/env python3
import numpy as np
import argparse
from dataloader import AntonymDataLoader
from svm import train_polynomial_svm, train_rbf_svm, compare_kernels, tune_svm_hyperparameters, evaluate_model_sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(X, y, title="Embeddings Visualization", method="pca"):
    """
    Visualize embeddings using dimensionality reduction.
    
    Parameters:
    X: Feature vectors
    y: Labels
    title: Plot title
    method: 'pca' or 'tsne'
    """
    # Use PCA or t-SNE for dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        method_name = "PCA"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method_name = "t-SNE"
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    labels = ['Non-Antonym', 'Antonym']
    
    for i, label in enumerate(labels):
        plt.scatter(
            X_reduced[y == i, 0], 
            X_reduced[y == i, 1],
            c=colors[i],
            label=label,
            alpha=0.7
        )
    
    plt.title(f"{title} ({method_name})")
    plt.legend()
    plt.savefig(f"{title.lower().replace(' ', '_')}_{method.lower()}.png")
    plt.close()
    print(f"Visualization saved as {title.lower().replace(' ', '_')}_{method.lower()}.png")

def main():
    parser = argparse.ArgumentParser(description='Antonym detection using SVM with Nomic embeddings')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--word_class', type=str, default='all', choices=['all', 'adjective', 'noun', 'verb'], 
                        help='Word class to use (default: all)')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--compare', action='store_true', help='Compare different kernels')
    parser.add_argument('--degree', type=int, default=4, help='Polynomial degree (default: 4)')
    parser.add_argument('--embedding_model', type=str, default='nomic-ai/nomic-embed-text-v1', 
                        help='Sentence transformer model to use (default: nomic-ai/nomic-embed-text-v1)')
    parser.add_argument('--no_embeddings', action='store_true', help='Use character n-grams instead of embeddings')
    parser.add_argument('--visualize', action='store_true', help='Visualize embeddings')
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} for {args.word_class} word class...")
    data_loader = AntonymDataLoader(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_numpy_data(word_class=args.word_class)
    
    # Reshape data for SVM
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_test = X_test.reshape(len(X_test), -1)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Optionally visualize embeddings
    if args.visualize and not args.no_embeddings:
        print("Visualizing embeddings...")
        # Combine train and test for better visualization
        X_combined = np.vstack([X_train[:500], X_test[:500]])
        y_combined = np.hstack([y_train[:500], y_test[:500]])
        
        visualize_embeddings(X_combined, y_combined, title="Antonym Embeddings", method="pca")
        visualize_embeddings(X_combined, y_combined, title="Antonym Embeddings", method="tsne")
    
    if args.tune:
        print("Tuning hyperparameters...")
        best_model = tune_svm_hyperparameters(X_train, y_train, X_val, y_val)
        print("\nEvaluating best model on test set:")
        evaluate_model_sklearn(best_model, X_test, y_test, model_name="Tuned SVM")
    
    elif args.compare:
        print("Comparing different kernels...")
        compare_kernels(X_train, y_train, X_test, y_test)
    
    else:
        # Default: train polynomial SVM with degree 4
        print(f"Training polynomial SVM with degree {args.degree}...")
        svm_clf = SVC(kernel='poly', degree=args.degree, verbose=True)
        svm_clf.fit(X_train, y_train)
        print("Polynomial SVM score:", svm_clf.score(X_test, y_test))
        evaluate_model_sklearn(svm_clf, X_test, y_test, model_name=f"Polynomial SVM (degree={args.degree})")

if __name__ == "__main__":
    main()