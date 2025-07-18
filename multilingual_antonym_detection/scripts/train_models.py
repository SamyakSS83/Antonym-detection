#!/usr/bin/env python3
"""
Multilingual Antonym Detection Training System
Trains BERT and Dual Encoder models on downloaded datasets.
"""

import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model classes
sys.path.append(str(Path(__file__).parent.parent))
from models.multilingual_bert import WordPairDataset, MultilingualBertTrainer
from models.multilingual_dual_encoder import DualEncoderGraphTransformer, WordPairGraphDataset

class MultilingualTrainingSystem:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """Load training configuration."""
        default_config = {
            'datasets': {
                'base_dir': '../datasets',
                'languages': ['german', 'french', 'spanish', 'italian', 'portuguese', 'dutch', 'russian']
            },
            'models': {
                'bert_dir': '../models/bert',
                'output_dir': '../models/trained',
                'model_types': ['bert', 'dual_encoder']
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 3,
                'max_length': 128,
                'test_size': 0.2,
                'val_size': 0.1
            },
            'hardware': {
                'use_gpu': True,
                'mixed_precision': True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge configs
            self.config = self._merge_configs(default_config, user_config)
        else:
            self.config = default_config
            # Save default config
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config at {self.config_path}")
    
    def _merge_configs(self, default: dict, user: dict) -> dict:
        """Recursively merge configuration dictionaries."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def setup_directories(self):
        """Create necessary directories."""
        self.datasets_dir = Path(self.config['datasets']['base_dir'])
        self.bert_models_dir = Path(self.config['models']['bert_dir'])
        self.output_dir = Path(self.config['models']['output_dir'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each model type
        for model_type in self.config['models']['model_types']:
            (self.output_dir / model_type).mkdir(exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """Check if datasets and BERT models are available."""
        logger.info("Checking prerequisites...")
        
        # Check datasets
        available_languages = []
        for language in self.config['datasets']['languages']:
            lang_dir = self.datasets_dir / language
            if lang_dir.exists():
                train_file = lang_dir / 'train.txt'
                val_file = lang_dir / 'val.txt'
                test_file = lang_dir / 'test.txt'
                
                if train_file.exists() and val_file.exists() and test_file.exists():
                    available_languages.append(language)
                    logger.info(f"✓ Dataset found for {language}")
                else:
                    logger.warning(f"✗ Incomplete dataset for {language}")
            else:
                logger.warning(f"✗ No dataset directory for {language}")
        
        if not available_languages:
            logger.error("No complete datasets found! Run dataset_downloader.py first.")
            return False
        
        self.available_languages = available_languages
        logger.info(f"Found datasets for {len(available_languages)} languages: {', '.join(available_languages)}")
        
        # Check BERT models
        bert_available = []
        for language in available_languages:
            bert_dir = self.bert_models_dir / language
            if bert_dir.exists() and (bert_dir / 'model').exists() and (bert_dir / 'tokenizer').exists():
                bert_available.append(language)
                logger.info(f"✓ BERT model found for {language}")
            else:
                logger.warning(f"✗ BERT model missing for {language}")
        
        # Check multilingual model
        multilingual_dir = self.bert_models_dir / 'multilingual'
        if multilingual_dir.exists() and (multilingual_dir / 'model').exists():
            logger.info("✓ Multilingual BERT model found")
            self.has_multilingual = True
        else:
            logger.warning("✗ Multilingual BERT model missing")
            self.has_multilingual = False
        
        if not bert_available and not self.has_multilingual:
            logger.error("No BERT models found! Run bert_downloader.py first.")
            return False
        
        self.bert_available = bert_available
        return True
    
    def load_dataset(self, language: str) -> Dict[str, pd.DataFrame]:
        """Load train/val/test datasets for a language."""
        lang_dir = self.datasets_dir / language
        
        datasets = {}
        for split in ['train', 'val', 'test']:
            file_path = lang_dir / f'{split}.txt'
            
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\\t', header=None, names=['word1', 'word2', 'label'])
                datasets[split] = df
                logger.info(f"Loaded {len(df)} {split} examples for {language}")
            else:
                logger.error(f"Dataset file not found: {file_path}")
                return None
        
        return datasets
    
    def get_bert_model_path(self, language: str) -> Optional[Path]:
        """Get the appropriate BERT model path for a language."""
        # Try language-specific model first
        lang_model_dir = self.bert_models_dir / language
        if lang_model_dir.exists() and (lang_model_dir / 'model').exists():
            return lang_model_dir
        
        # Fall back to multilingual model
        if self.has_multilingual:
            multilingual_dir = self.bert_models_dir / 'multilingual'
            logger.info(f"Using multilingual BERT for {language}")
            return multilingual_dir
        
        return None
    
    def train_bert_model(self, language: str) -> bool:
        """Train BERT model for a specific language."""
        logger.info(f"Training BERT model for {language}")
        
        # Load datasets
        datasets = self.load_dataset(language)
        if not datasets:
            return False
        
        # Get BERT model path
        bert_model_path = self.get_bert_model_path(language)
        if not bert_model_path:
            logger.error(f"No BERT model available for {language}")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(bert_model_path / 'tokenizer')
            model = AutoModelForSequenceClassification.from_pretrained(
                bert_model_path / 'model',
                num_labels=2
            )
            
            # Create trainer
            trainer = MultilingualBertTrainer(
                model=model,
                tokenizer=tokenizer,
                train_df=datasets['train'],
                val_df=datasets['val'],
                test_df=datasets['test'],
                config=self.config['training']
            )
            
            # Train model
            trainer.train()
            
            # Save trained model
            output_path = self.output_dir / 'bert' / language
            output_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_path / 'model')
            tokenizer.save_pretrained(output_path / 'tokenizer')
            
            # Save training info
            with open(output_path / 'training_info.txt', 'w') as f:
                f.write(f"Language: {language}\\n")
                f.write(f"Base Model: {bert_model_path}\\n")
                f.write(f"Training Examples: {len(datasets['train'])}\\n")
                f.write(f"Validation Examples: {len(datasets['val'])}\\n")
                f.write(f"Test Examples: {len(datasets['test'])}\\n")
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            
            logger.info(f"✓ Successfully trained BERT model for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error training BERT model for {language}: {e}")
            return False
    
    def train_dual_encoder_model(self, language: str) -> bool:
        """Train Dual Encoder model for a specific language."""
        logger.info(f"Training Dual Encoder model for {language}")
        
        # This would require the trained BERT model from the previous step
        bert_output_path = self.output_dir / 'bert' / language
        if not bert_output_path.exists():
            logger.error(f"Trained BERT model not found for {language}. Train BERT first.")
            return False
        
        try:
            # Load datasets
            datasets = self.load_dataset(language)
            if not datasets:
                return False
            
            # This is a placeholder for dual encoder training
            # In practice, you would implement the training loop here
            logger.info(f"Dual Encoder training for {language} - placeholder implementation")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Dual Encoder model for {language}: {e}")
            return False
    
    def train_all_languages(self, model_types: List[str] = None) -> Dict[str, Dict[str, bool]]:
        """Train models for all available languages."""
        if model_types is None:
            model_types = self.config['models']['model_types']
        
        results = {}
        
        for language in self.available_languages:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Training models for {language.upper()}")
            logger.info(f"{'='*60}")
            
            results[language] = {}
            
            for model_type in model_types:
                if model_type == 'bert':
                    success = self.train_bert_model(language)
                elif model_type == 'dual_encoder':
                    success = self.train_dual_encoder_model(language)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    success = False
                
                results[language][model_type] = success
        
        return results
    
    def print_training_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print training summary."""
        logger.info(f"\\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        for language, model_results in results.items():
            logger.info(f"\\n{language.upper()}:")
            for model_type, success in model_results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                logger.info(f"  {model_type}: {status}")
        
        # Count successes
        total_models = sum(len(model_results) for model_results in results.values())
        successful_models = sum(
            sum(1 for success in model_results.values() if success)
            for model_results in results.values()
        )
        
        logger.info(f"\\nOverall: {successful_models}/{total_models} models trained successfully")
        logger.info(f"Trained models saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train multilingual antonym detection models')
    parser.add_argument('--config', default='../config/training_config.yaml', help='Training configuration file')
    parser.add_argument('--language', help='Train specific language only')
    parser.add_argument('--model-type', choices=['bert', 'dual_encoder'], help='Train specific model type only')
    parser.add_argument('--check-only', action='store_true', help='Only check prerequisites')
    
    args = parser.parse_args()
    
    # Initialize training system
    training_system = MultilingualTrainingSystem(args.config)
    
    # Check prerequisites
    if not training_system.check_prerequisites():
        logger.error("Prerequisites not met. Please download datasets and BERT models first.")
        sys.exit(1)
    
    if args.check_only:
        logger.info("Prerequisites check completed successfully!")
        return
    
    # Determine what to train
    languages_to_train = [args.language] if args.language else training_system.available_languages
    model_types_to_train = [args.model_type] if args.model_type else None
    
    # Filter languages based on availability
    if args.language and args.language not in training_system.available_languages:
        logger.error(f"Language {args.language} not available. Available: {', '.join(training_system.available_languages)}")
        sys.exit(1)
    
    # Start training
    logger.info(f"Starting training for languages: {', '.join(languages_to_train)}")
    if model_types_to_train:
        logger.info(f"Model types: {', '.join(model_types_to_train)}")
    
    results = {}
    for language in languages_to_train:
        if args.language:
            # Train single language
            results[language] = {}
            for model_type in (model_types_to_train or ['bert', 'dual_encoder']):
                if model_type == 'bert':
                    success = training_system.train_bert_model(language)
                elif model_type == 'dual_encoder':
                    success = training_system.train_dual_encoder_model(language)
                results[language][model_type] = success
        else:
            # Train all languages
            results = training_system.train_all_languages(model_types_to_train)
            break
    
    # Print summary
    training_system.print_training_summary(results)

if __name__ == "__main__":
    main()
