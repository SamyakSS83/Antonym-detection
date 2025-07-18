#!/usr/bin/env python3
"""
Fallback WordNet downloader that works around SQLite compatibility issues.
Downloads WordNet data files directly and parses them manually.
"""

import sys
import requests
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import gzip
import re
from typing import Set, Tuple, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FallbackWordNetDownloader:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Direct WordNet tab-separated data URLs from OMW
        self.wordnet_urls = {
            'french': [
                'https://github.com/omwn/omw-data/raw/main/wns/fra/wn-data-fra.tab'
            ],
            'italian': [
                'https://github.com/omwn/omw-data/raw/main/wns/ita/wn-data-ita.tab',
                'https://github.com/omwn/omw-data/raw/main/wns/iwn/wn-data-ita.tab'
            ],
            'portuguese': [
                'https://github.com/omwn/omw-data/raw/main/wns/por/wn-data-por.tab'
            ],
            'dutch': [
                'https://github.com/omwn/omw-data/raw/main/wns/nld/wn-data-nld.tab'
            ],
            'spanish': [
                'https://github.com/omwn/omw-data/raw/main/wns/mcr/wn-data-spa.tab'
            ],
            'russian': [
                'https://github.com/omwn/omw-data/raw/main/wns/wikt/wn-wikt-rus.tab'
            ]
        }
        
        # Language codes for cleanup
        self.lang_codes = {
            'german': 'de',
            'french': 'fr',
            'italian': 'it',
            'portuguese': 'pt',
            'dutch': 'nl',
            'spanish': 'es',
            'russian': 'ru'
        }
    
    def download_wordnet_tab(self, language: str) -> str:
        """Download WordNet tab-separated file for a language."""
        if language not in self.wordnet_urls:
            logger.warning(f"No WordNet URLs configured for {language}")
            return None
        
        logger.info(f"Downloading WordNet tab data for {language}")
        
        # Try each URL until one works
        for i, url in enumerate(self.wordnet_urls[language]):
            try:
                logger.info(f"Trying URL {i+1}/{len(self.wordnet_urls[language])}: {url}")
                response = requests.get(url, timeout=60)
                
                if response.status_code == 200:
                    tab_content = response.text
                    if tab_content and len(tab_content) > 1000:  # Basic validity check
                        logger.info(f"Successfully downloaded WordNet tab data for {language} ({len(tab_content)} characters)")
                        return tab_content
                    else:
                        logger.warning(f"Downloaded tab data seems too small: {len(tab_content) if tab_content else 0} characters")
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue
        
        logger.error(f"Failed to download WordNet tab data for {language}")
        return None
    
    def parse_wordnet_tab_for_antonyms(self, tab_content: str, language: str) -> Set[Tuple[str, str]]:
        """Parse WordNet tab-separated file to extract potential antonym pairs."""
        antonym_pairs = set()
        
        try:
            logger.info(f"Parsing WordNet tab data for {language}")
            
            # Parse tab-separated format
            # Format: offset-pos<tab>type<tab>lemma
            # We'll look for words in the same synset and use external knowledge for antonyms
            
            synset_words = {}  # synset_id -> list of words
            all_words = set()
            
            lines = tab_content.strip().split('\n')
            logger.info(f"Processing {len(lines)} lines from WordNet tab data")
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    synset_id = parts[0]  # e.g., "01234567-n"
                    rel_type = parts[1]   # e.g., "fra:lemma"
                    lemma = parts[2].lower().strip()
                    
                    # Only process lemma relations
                    if ':lemma' in rel_type and lemma:
                        word_clean = self._clean_word(lemma, self.lang_codes[language])
                        if word_clean and len(word_clean) >= 3:
                            if synset_id not in synset_words:
                                synset_words[synset_id] = []
                            synset_words[synset_id].append(word_clean)
                            all_words.add(word_clean)
            
            logger.info(f"Found {len(synset_words)} synsets with {len(all_words)} unique words")
            
            # For a simple approach, we'll use common antonym patterns
            # This is a basic heuristic since the tab files don't contain explicit antonym relations
            antonym_prefixes = {
                'fr': ['in', 'im', 'ir', 'il', 'non', 'dé', 'des', 'anti', 'contre'],
                'es': ['in', 'im', 'ir', 'des', 'anti', 'contra', 'no'],
                'it': ['in', 'im', 'ir', 'dis', 'anti', 'non', 'contro'],
                'pt': ['in', 'im', 'ir', 'des', 'anti', 'contra', 'não'],
                'nl': ['on', 'in', 'niet', 'anti', 'tegen'],
                'ru': ['не', 'без', 'анти', 'противо']
            }
            
            lang_code = self.lang_codes[language]
            prefixes = antonym_prefixes.get(lang_code, ['in', 'un', 'non', 'anti'])
            
            # Simple heuristic: find potential antonym pairs based on prefixes
            word_list = list(all_words)
            for i, word1 in enumerate(word_list):
                for prefix in prefixes:
                    # Check if removing prefix gives us another word
                    if word1.startswith(prefix) and len(word1) > len(prefix) + 2:
                        potential_antonym = word1[len(prefix):]
                        if potential_antonym in all_words:
                            pair = tuple(sorted([word1, potential_antonym]))
                            antonym_pairs.add(pair)
                    
                    # Check if adding prefix gives us another word
                    potential_antonym = prefix + word1
                    if potential_antonym in all_words:
                        pair = tuple(sorted([word1, potential_antonym]))
                        antonym_pairs.add(pair)
            
            logger.info(f"Extracted {len(antonym_pairs)} potential antonym pairs using prefix heuristics")
            return antonym_pairs
            
        except Exception as e:
            logger.error(f"Error parsing WordNet tab data for {language}: {e}")
            return set()
    
    def _get_synset_words(self, synset, lexical_entries) -> List[str]:
        """Get all words from a synset."""
        words = []
        
        # Look for direct word references in synset
        for member in synset.findall('.//Member'):
            word_ref = member.get('targets', '')
            if word_ref in lexical_entries:
                lemma = lexical_entries[word_ref].find('Lemma')
                if lemma is not None:
                    word = lemma.get('writtenForm', '').lower().strip()
                    if word and len(word) >= 3:
                        words.append(word)
        
        return words
    
    def _clean_word(self, word: str, lang_code: str) -> str:
        """Clean word based on language-specific character sets."""
        if lang_code == 'de':
            cleaned = re.sub(r'[^a-zA-ZäöüßÄÖÜ]', '', word)
        elif lang_code == 'es':
            cleaned = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ]', '', word)
        elif lang_code == 'fr':
            cleaned = re.sub(r'[^a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', '', word)
        elif lang_code == 'it':
            cleaned = re.sub(r'[^a-zA-Zàèéìíîòóù]', '', word)
        elif lang_code == 'pt':
            cleaned = re.sub(r'[^a-zA-ZáâãàéêíóôõúçÁÂÃÀÉÊÍÓÔÕÚÇ]', '', word)
        elif lang_code == 'nl':
            cleaned = re.sub(r'[^a-zA-Zëüöï]', '', word)
        elif lang_code == 'ru':
            cleaned = re.sub(r'[^а-яёА-ЯЁ]', '', word)
        else:
            cleaned = re.sub(r'[^a-zA-Z]', '', word)
        
        return cleaned if len(cleaned) >= 3 else ''
    
    def download_conceptnet_data(self, language: str) -> Set[Tuple[str, str]]:
        """Download ConceptNet data (same as professional downloader)."""
        try:
            import requests
            import time
            
            lang_code = self.lang_codes.get(language)
            if not lang_code:
                return set()
            
            logger.info(f"Downloading ConceptNet data for {language} ({lang_code})")
            
            antonym_pairs = set()
            offset = 0
            limit = 1000
            
            while offset < 10000:
                url = f"http://api.conceptnet.io/query?node=/c/{lang_code}&rel=/r/Antonym&limit={limit}&offset={offset}"
                
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    edges = data.get('edges', [])
                    if not edges:
                        break
                    
                    for edge in edges:
                        start_info = edge.get('start', {})
                        end_info = edge.get('end', {})
                        
                        start_label = start_info.get('label', '').strip()
                        end_label = end_info.get('label', '').strip()
                        start_lang = start_info.get('language', '')
                        end_lang = end_info.get('language', '')
                        
                        if (start_label and end_label and 
                            start_lang == lang_code and end_lang == lang_code and
                            len(start_label.split()) == 1 and len(end_label.split()) == 1):
                            
                            word1_clean = self._clean_word(start_label.lower(), lang_code)
                            word2_clean = self._clean_word(end_label.lower(), lang_code)
                            
                            if (word1_clean and word2_clean and 
                                word1_clean != word2_clean and
                                len(word1_clean) > 2 and len(word2_clean) > 2):
                                pair = tuple(sorted([word1_clean, word2_clean]))
                                antonym_pairs.add(pair)
                    
                    offset += limit
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error downloading ConceptNet data: {e}")
                    break
            
            logger.info(f"Downloaded {len(antonym_pairs)} antonym pairs from ConceptNet")
            return antonym_pairs
            
        except Exception as e:
            logger.error(f"Error downloading ConceptNet data: {e}")
            return set()
    
    def process_language(self, language: str) -> bool:
        """Process a complete language dataset using fallback methods."""
        logger.info(f"Starting fallback dataset processing for {language}")
        
        # Create language directory
        lang_dir = self.output_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Step 1: Try to download WordNet tab data
        wordnet_pairs = set()
        tab_content = self.download_wordnet_tab(language)
        if tab_content:
            wordnet_pairs = self.parse_wordnet_tab_for_antonyms(tab_content, language)
        
        # Step 2: Download ConceptNet data
        conceptnet_pairs = self.download_conceptnet_data(language)
        
        # Step 3: Combine all sources
        all_pairs = wordnet_pairs.union(conceptnet_pairs)
        
        if not all_pairs:
            logger.warning(f"No antonym pairs found for {language}")
            return False
        
        # Step 4: Save datasets
        self._save_datasets(language, {
            'wordnet': list(wordnet_pairs),
            'conceptnet': list(conceptnet_pairs),
            'combined': list(all_pairs)
        })
        
        logger.info(f"Completed fallback processing for {language}: {len(all_pairs)} total pairs")
        return True
    
    def _save_datasets(self, language: str, datasets: Dict[str, List[Tuple[str, str]]]):
        """Save all datasets with proper splits."""
        import random
        import time
        
        lang_dir = self.output_dir / language
        
        # Save individual sources
        for source_name, pairs in datasets.items():
            if pairs:
                source_file = lang_dir / f"{source_name}_antonyms.txt"
                with open(source_file, 'w', encoding='utf-8') as f:
                    for word1, word2 in pairs:
                        f.write(f"{word1}\t{word2}\t1\n")
                logger.info(f"Saved {len(pairs)} pairs to {source_file}")
        
        # Create train/val/test splits from combined data
        combined_pairs = datasets.get('combined', [])
        if combined_pairs:
            random.shuffle(combined_pairs)
            
            total = len(combined_pairs)
            train_size = int(total * 0.7)
            val_size = int(total * 0.15)
            
            train_pairs = combined_pairs[:train_size]
            val_pairs = combined_pairs[train_size:train_size + val_size]
            test_pairs = combined_pairs[train_size + val_size:]
            
            # Save splits
            for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
                split_file = lang_dir / f"{split_name}.txt"
                with open(split_file, 'w', encoding='utf-8') as f:
                    for word1, word2 in split_pairs:
                        f.write(f"{word1}\t{word2}\t1\n")
            
            # Save statistics
            stats_file = lang_dir / "statistics.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Language: {language}\n")
                f.write(f"Total unique pairs: {total}\n")
                f.write(f"Train pairs: {len(train_pairs)}\n")
                f.write(f"Val pairs: {len(val_pairs)}\n")
                f.write(f"Test pairs: {len(test_pairs)}\n")
                f.write(f"WordNet pairs: {len(datasets.get('wordnet', []))}\n")
                f.write(f"ConceptNet pairs: {len(datasets.get('conceptnet', []))}\n")
                f.write(f"Sources: Fallback WordNet XML + ConceptNet\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Created splits for {language}: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fallback WordNet-based multilingual antonym dataset downloader')
    parser.add_argument('--output', default='../datasets', help='Output directory')
    parser.add_argument('--language', help='Specific language to download (default: all)')
    
    args = parser.parse_args()
    
    downloader = FallbackWordNetDownloader(args.output)
    
    # Determine languages to process
    supported_languages = ['german', 'french', 'spanish', 'italian', 'portuguese', 'dutch', 'russian']
    
    if args.language:
        if args.language not in supported_languages:
            logger.error(f"Unsupported language: {args.language}")
            logger.info(f"Supported: {', '.join(supported_languages)}")
            sys.exit(1)
        languages = [args.language]
    else:
        languages = supported_languages
    
    # Process datasets
    logger.info("Starting fallback multilingual dataset download using direct WordNet XML + ConceptNet")
    
    success_count = 0
    for language in languages:
        try:
            if downloader.process_language(language):
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing {language}: {e}")
            continue
    
    logger.info(f"Fallback dataset download completed! Successfully processed {success_count}/{len(languages)} languages")
    logger.info(f"Results saved to: {downloader.output_dir}")

if __name__ == "__main__":
    main()
