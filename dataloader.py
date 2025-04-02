import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
import torch
from sentence_transformers import SentenceTransformer

class AntonymDataLoader:
    def __init__(self, data_dir, embedding_model="nomic-ai/nomic-embed-text-v1", use_embeddings=True):
        """
        Initialize the data loader for antonym detection.
        
        Parameters:
        data_dir (str): Directory containing the dataset files
        embedding_model (str): Name or path of embedding model to use
        use_embeddings (bool): Whether to use pre-trained embeddings or bag-of-words
        """
        self.data_dir = data_dir
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        if use_embeddings:
            try:
                print(f"Loading embedding model: {embedding_model}...")
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Successfully loaded embedding model")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Falling back to bag-of-words representation")
                self.use_embeddings = False
        
        if not self.use_embeddings:
            self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
    
    def load_dataset(self, dataset_type, word_class="all"):
        """
        Load a specific dataset.
        
        Parameters:
        dataset_type (str): 'train', 'val', or 'test'
        word_class (str): 'adjective', 'noun', 'verb', or 'all'
        
        Returns:
        tuple: (X, y) where X is the features and y is the labels
        """
        if word_class == "all":
            word_classes = ["adjective", "noun", "verb"]
        else:
            word_classes = [word_class]
        
        all_pairs = []
        all_labels = []
        
        for wc in word_classes:
            filename = f"{wc}-pairs.{dataset_type}"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist")
                continue
                
            print(f"Loading {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word1, word2, label = parts[0], parts[1], int(parts[2])
                        all_pairs.append((word1, word2))
                        all_labels.append(label)
        
        if not all_pairs:
            raise ValueError(f"No data found for {dataset_type} in {word_class}")
        
        X = self._featurize_pairs(all_pairs)
        y = np.array(all_labels)
        
        return X, y
    
    def _featurize_pairs(self, word_pairs):
        """
        Convert word pairs to feature vectors.
        
        Parameters:
        word_pairs (list): List of (word1, word2) tuples
        
        Returns:
        numpy.ndarray: Feature vectors
        """
        if self.use_embeddings and self.embedding_model:
            return self._nomic_embeddings_features(word_pairs)
        else:
            return self._bow_features(word_pairs)
    
    def _nomic_embeddings_features(self, word_pairs):
        """
        Create features using Nomic embeddings from sentence-transformers.
        
        Parameters:
        word_pairs (list): List of (word1, word2) tuples
        
        Returns:
        numpy.ndarray: Feature vectors
        """
        print(f"Creating features using {self.embedding_model_name} embeddings...")
        features = []
        
        # Get all unique words
        all_words = []
        for word1, word2 in word_pairs:
            all_words.append(word1)
            all_words.append(word2)
        unique_words = list(set(all_words))
        
        # Generate embeddings for all unique words at once (for efficiency)
        print(f"Generating embeddings for {len(unique_words)} unique words...")
        embeddings_dict = {}
        
        # Process in batches to avoid memory issues
        batch_size = 128
        for i in range(0, len(unique_words), batch_size):
            batch_words = unique_words[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_words, show_progress_bar=False)
            for word, embedding in zip(batch_words, batch_embeddings):
                embeddings_dict[word] = embedding
        
        print("Combining features for word pairs...")
        for word1, word2 in word_pairs:
            vec1 = embeddings_dict[word1]
            vec2 = embeddings_dict[word2]
            
            # Feature vector: concatenation of both word vectors, their difference, and their element-wise product
            diff = np.abs(vec1 - vec2)
            prod = vec1 * vec2
            
            # Can also include cosine similarity as a feature
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Create the combined feature vector
            combined = np.concatenate([vec1, vec2, diff, prod, [sim]])
            features.append(combined)
        
        return np.array(features)
    
    def _bow_features(self, word_pairs):
        """
        Create features using bag-of-character-ngrams representation.
        
        Parameters:
        word_pairs (list): List of (word1, word2) tuples
        
        Returns:
        numpy.ndarray: Feature vectors
        """
        # Prepare all words for vectorization
        all_words = []
        for word1, word2 in word_pairs:
            all_words.append(word1)
            all_words.append(word2)
        
        # Fit the vectorizer if it hasn't been fit yet
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(all_words)
        
        features = []
        for word1, word2 in word_pairs:
            vec1 = self.vectorizer.transform([word1]).toarray().flatten()
            vec2 = self.vectorizer.transform([word2]).toarray().flatten()
            
            # Combine features
            combined = np.concatenate([vec1, vec2, np.abs(vec1 - vec2), vec1 * vec2])
            features.append(combined)
        
        return np.array(features)
    
    def get_data_loaders(self, word_class="all", batch_size=32):
        """
        Get PyTorch DataLoader objects for train, validation, and test sets.
        
        Parameters:
        word_class (str): 'adjective', 'noun', 'verb', or 'all'
        batch_size (int): Batch size for DataLoader
        
        Returns:
        tuple: (train_loader, val_loader, test_loader)
        """
        X_train, y_train = self.load_dataset('train', word_class)
        X_val, y_val = self.load_dataset('val', word_class)
        X_test, y_test = self.load_dataset('test', word_class)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def get_numpy_data(self, word_class="all"):
        """
        Get numpy arrays for train, validation, and test sets.
        
        Parameters:
        word_class (str): 'adjective', 'noun', 'verb', or 'all'
        
        Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X_train, y_train = self.load_dataset('train', word_class)
        X_val, y_val = self.load_dataset('val', word_class)
        X_test, y_test = self.load_dataset('test', word_class)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_embedding_dimension(self):
        """
        Get the dimension of the feature vectors.
        
        Returns:
        int: Dimension of feature vectors
        """
        if self.use_embeddings and self.embedding_model:
            # Create a sample embedding to get the dimension
            sample_embedding = self.embedding_model.encode("sample")
            # Factor of 4 because we concatenate original vectors, difference, product
            # Plus 1 for cosine similarity
            return sample_embedding.shape[0] * 4 + 1
        elif hasattr(self.vectorizer, 'vocabulary_'):
            vocab_size = len(self.vectorizer.vocabulary_)
            # Factor of 4 because we concatenate original vectors, difference, product
            return vocab_size * 4
        else:
            return None


if __name__ == "__main__":
    # Example usage
    data_loader = AntonymDataLoader(data_dir="dataset", embedding_model="nomic-ai/nomic-embed-text-v1")
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_numpy_data()
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")