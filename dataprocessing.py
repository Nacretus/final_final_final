import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from CONFIG import*
# =============================================================================
# Hybrid Character Vocabulary
# =============================================================================

class HybridCharacterVocabulary:
    def __init__(self, fixed_alphabet=None, max_vocab_size=500):
        # Default alphabet if none provided
        self.default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        # Initialize with special tokens
        self.char_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_char = {0: self.pad_token, 1: self.unk_token}
        self.n_chars = 2  # Count of special tokens

        # Character frequency tracking
        self.char_count = {}

        # Add fixed alphabet first if provided
        if fixed_alphabet is not None:
            self.add_fixed_alphabet(fixed_alphabet)
    
    def add_fixed_alphabet(self, alphabet):
        """Add a fixed alphabet to the vocabulary."""
        print(f"Adding fixed alphabet with {len(alphabet)} characters")
        
        # Add each character from the alphabet to the vocabulary
        for char in alphabet:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.char_count[char] = float('inf')  # Mark as fixed alphabet character
                self.n_chars += 1

        print(f"After adding fixed alphabet: {self.n_chars} characters")
    
    def build_from_texts(self, texts, min_count=2):
        """Build additional vocabulary from texts."""
        print("Extending vocabulary from training data...")
        
        # Count characters
        for text in texts:
            for char in text:
                if char not in self.char_count:
                    self.char_count[char] = 0
                self.char_count[char] += 1

        # Add frequently occurring characters that aren't already in the vocabulary
        chars_added = 0
        for char, count in sorted(self.char_count.items(), key=lambda x: x[1], reverse=True):
            # Skip if already in vocabulary
            if char in self.char_to_idx:
                continue
                
            # Skip if below minimum count
            if count < min_count:
                continue
                
            # Skip if we've reached maximum vocabulary size
            if self.n_chars >= self.max_vocab_size:
                break
                
            # Add to vocabulary
            self.char_to_idx[char] = self.n_chars
            self.idx_to_char[self.n_chars] = char
            self.n_chars += 1
            chars_added += 1
        
        # Add special attention to important characters for toxicity detection
        special_toxicity_chars = [
            # Common substitutions in toxic text
            '@', '0', '1', '3', '4', '$']

        for char in special_toxicity_chars:
            if char not in self.char_to_idx and self.n_chars < self.max_vocab_size:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.n_chars += 1
                chars_added += 1
            
        print(f"Added {chars_added} new characters from training data")
        print(f"Final vocabulary size: {self.n_chars} characters")
        
        # Print some statistics about character coverage
        total_chars = sum(self.char_count.values())
        covered_chars = sum(count for char, count in self.char_count.items() if char in self.char_to_idx)
        coverage = covered_chars / total_chars * 100 if total_chars > 0 else 0
        print(f"Character coverage: {coverage:.2f}% of all character occurrences")

    def encode_text(self, text, max_len=300):
        """Convert text to sequence of character indices."""
        # Pre-allocate array with pad tokens
        indices = np.full(max_len, self.char_to_idx[self.pad_token], dtype=np.int64)

        # Fill with actual character indices
        for i, char in enumerate(text[:max_len]):
            indices[i] = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])

        return indices

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# =============================================================================
# Text Preprocessing
# =============================================================================

# Pre-compiled patterns for efficiency
WHITESPACE_PATTERN = re.compile(r'\s+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
def preprocess_text(text, max_len=300):
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with special token
    text = URL_PATTERN.sub(" <URL> ", text)
       
    # Remove excessive whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Truncate if needed
    if len(text) > max_len:
        text = text[:max_len]
    
    return text

# Optional: Language detection
def detect_language(text):
    # Common Tagalog words
    tagalog_markers = [
    # Standard function words / particles and common pronouns
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila", 
    "ang", "ng", "sa", "mga", "ni", "namin", "natin", "nila",
    "hindi", "oo", "opo", "wala", "meron", "dahil", "kasi",
    
    # Additional function words, particles, and connectors
    "na", "nang", "lang", "lamang", "ba", "daw", "raw", "pala",
    "kaya", "pero", "ngunit", "subalit", "at", "o", "kung", 
    "kapag", "pag", "sapagkat", "para", "pwede", "puwede", 
    "baka", "siguro", "marahil", "naman", "nga", "kay", "kina", "nina",
    
    # Pronouns, demonstratives, and interrogatives
    "ito", "iyan", "iyon", 
    "sino", "ano", "saan", "kailan", "bakit", "paano", "ilan",
    
    # Common verbal affixes (use with caution for token matching)
    "mag", "nag", "um", "in", "an", "ma", "ipag", "ipa", "pa",
    
    # Additional words and expressions
    "ayaw", "paki", "salamat", "walang", "anuman", 
    "pasensya", "pasensiya", "mahal", "murang", 
    "malaki", "maliit", "masaya", "malungkot", 
    "maganda", "gwapo",
    
    # ADDED: More common Tagalog words to improve detection
    "yung", "po", "opo", "yun", "dito", "diyan", "doon",
    "kanina", "bukas", "kahapon", "ngayon", "mamaya",
    "nasaan", "nasaaan", "gusto", "ayoko", "talaga",
    "sobra", "grabe", "mabuti", "masama"
    ]
    
    # Clean and tokenize the text
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Skip very short texts
    if len(words) < 3:
        return 'en'  # Default to English
    
    # Count Tagalog markers
    tagalog_count = sum(1 for word in words if word in tagalog_markers)
    tagalog_ratio = tagalog_count / len(words)
    
    # UPDATED: Lowered threshold for Tagalog detection to be more sensitive
    if tagalog_ratio > 0.12:  # If more than 12% words are Tagalog markers
        return 'tl'
    else:
        return 'en'

def extract_toxicity_features(text):
    """Extract additional features for toxicity detection."""
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
        
    features = {}
    
    # 1. Check for ALL CAPS segments
    words = text.split()
    all_caps_words = [w for w in words if len(w) > 2 and w.isupper()]
    features['all_caps_ratio'] = len(all_caps_words) / max(1, len(words))
    
    # 2. Toxic keyword checking
    # Read toxic keywords from CSV file - do this once and cache
    # Use global variable to avoid reading file for every function call
    global toxic_keywords
    if 'toxic_keywords' not in globals():
        import pandas as pd
        import os
        
        # Read keywords from CSV
        csv_path = 'extended_profanity_list.csv'
        if os.path.exists(csv_path):
            try:
                # Try different ways to read the CSV depending on its structure
                try:
                    # If the CSV has headers
                    df = pd.read_csv(csv_path)
                    # Assuming the first column contains the keywords
                    toxic_keywords = df.iloc[:, 0].tolist()
                except:
                    # If the CSV is just a list of words with no header
                    toxic_keywords = pd.read_csv(csv_path, header=None)[0].tolist()
                
                # Remove any NaN values and convert to lowercase
                toxic_keywords = [str(word).lower() for word in toxic_keywords if str(word) != 'nan']
                print(f"Loaded {len(toxic_keywords)} toxic keywords from {csv_path}")
            except Exception as e:
                print(f"Error loading toxic keywords from CSV: {e}")
                # Fallback to a small default list
                toxic_keywords = ['fuck', 'shit', 'ass', 'bitch', 'damn', 'cunt', 'dick', 'pussy', 'nigger', 'faggot']
        else:
            print(f"Warning: Toxic keyword file {csv_path} not found. Using default keywords.")
            # Fallback to a small default list
            toxic_keywords = ['fuck', 'shit', 'ass', 'bitch', 'damn', 'cunt', 'dick', 'pussy', 'nigger', 'faggot']
    
    # Count toxic keywords (case insensitive)
    lower_text = text.lower()
    keyword_count = 0
    detected_keywords = []
    
    for keyword in toxic_keywords:
        if keyword in lower_text:
            keyword_count += 1
            detected_keywords.append(keyword)
    
    features['toxic_keyword_count'] = keyword_count
    features['toxic_keyword_ratio'] = keyword_count / max(1, len(words))
    features['detected_keywords'] = detected_keywords
    
    # 3. Safe word detection
    # Get safe word settings from CONFIG
    global safe_words
    if 'safe_words' not in globals():
        try:
            from CONFIG import SAFE_WORD_SETTINGS
            safe_words = SAFE_WORD_SETTINGS.get('benign_phrases', [])
            print(f"Using {len(safe_words)} safe words/phrases from configuration")
        except ImportError:
            # Fallback if CONFIG isn't available
            safe_words = []
            print("Warning: Could not import SAFE_WORD_SETTINGS, safe word detection disabled")
    
    # Check for safe words/phrases
    safe_word_count = 0
    detected_safe_words = []
    
    for safe_phrase in safe_words:
        if safe_phrase.lower() in lower_text:
            safe_word_count += 1
            detected_safe_words.append(safe_phrase)
    
    features['safe_word_count'] = safe_word_count
    features['safe_word_ratio'] = safe_word_count / max(1, len(words))
    features['detected_safe_words'] = detected_safe_words
    
    return features

# =============================================================================
# Dataset and DataLoader
# =============================================================================

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, char_vocab=None, max_len=300, detect_lang=False):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.detect_lang = detect_lang
        
        # Pre-process all texts
        self.processed_texts = [preprocess_text(text, max_len) for text in texts]
        
        # Extract toxicity features for all texts
        print("Extracting toxicity features...")
        self.toxicity_features = [extract_toxicity_features(text) for text in self.processed_texts]
        
        # Initialize character vocabulary if not provided
        if char_vocab is None:
            if CONFIG.get('use_hybrid_vocabulary', True):
                self.char_vocab = HybridCharacterVocabulary(
                    fixed_alphabet=CONFIG.get('alphabet', None),
                    max_vocab_size=CONFIG.get('max_vocab_size', 500)
                )
                self.char_vocab.build_from_texts(self.processed_texts, min_count=2)
            else:
                self.char_vocab = HybridCharacterVocabulary()
                self.char_vocab.build_vocab(self.processed_texts)
        else:
            self.char_vocab = char_vocab
        
        # Detect languages if enabled
        if self.detect_lang:
            print("Detecting languages for texts...")
            self.languages = [detect_language(text) for text in self.processed_texts]
            lang_counts = Counter(self.languages)
            print(f"Language distribution: {dict(lang_counts)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert idx to int if it's a string
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                raise TypeError(f"Cannot convert idx '{idx}' to integer")
            
        # Get pre-processed text
        processed_text = self.processed_texts[idx]
            
        # Get toxicity features
        features = self.toxicity_features[idx]
            
        # Encode text to character indices
        char_ids = self.char_vocab.encode_text(processed_text, self.max_len)
            
        if self.labels is not None:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.float),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float),
                # Add safe word features
                'safe_word_count': torch.tensor(features.get('safe_word_count', 0), dtype=torch.float),
                'safe_word_ratio': torch.tensor(features.get('safe_word_ratio', 0), dtype=torch.float)
            }
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item
        else:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float),
                # Add safe word features
                'safe_word_count': torch.tensor(features.get('safe_word_count', 0), dtype=torch.float),
                'safe_word_ratio': torch.tensor(features.get('safe_word_ratio', 0), dtype=torch.float)
            }
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data_from_csv(file_path, text_column=None, toxicity_column=None, category_columns=None):
    print(f"Loading data from {file_path}...")
    
    # Get column names from CONFIG if not specified
    if text_column is None:
        text_column = CONFIG.get('text_column', 'text')
    if toxicity_column is None:
        toxicity_column = CONFIG.get('toxicity_column', 'toxicity_level')
    if category_columns is None:
        category_columns = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    
    try:
        # Try with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(file_path, encoding='latin-1')
        
        print(f"Loaded {len(df)} rows from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    # Check if columns exist
    missing_columns = []
    if text_column not in df.columns:
        missing_columns.append(text_column)
    if toxicity_column not in df.columns:
        missing_columns.append(toxicity_column)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Extract texts
    texts = df[text_column].tolist()
    
    # Create labels array [toxicity, insult, profanity, threat, identity_hate]
    toxicity_levels = df[toxicity_column].astype(int).values
    
    # Initialize the labels array
    labels = np.zeros((len(df), 1 + len(category_columns)))
    labels[:, 0] = toxicity_levels
    
    # Add category values if available
    for i, col in enumerate(category_columns):
        if col in df.columns:
            labels[:, i+1] = df[col].astype(int).values
        else:
            print(f"Warning: Category column '{col}' not found. Using all zeros.")
    
    # Print data distribution
    print(f"\nToxicity level distribution:")
    for level, count in sorted(Counter(toxicity_levels).items()):
        percentage = count / len(toxicity_levels) * 100
        print(f"  Level {level}: {count} examples ({percentage:.1f}%)")
    
    print(f"\nCategory distribution:")
    for i, col in enumerate(category_columns):
        positive_count = np.sum(labels[:, i+1] == 1)
        percentage = positive_count / len(labels) * 100
        print(f"  {col}: {positive_count} positive examples ({percentage:.1f}%)")
    
    return texts, labels

def create_data_loaders(texts, labels, char_vocab=None, test_size=0.2, val_size=0.25, 
                        batch_size=32, num_workers=4, max_len=300, detect_lang=False, seed=42):

    # Perform train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels[:, 0]
    )
    
    # Perform train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=seed, stratify=train_labels[:, 0]
    )
    
    print(f"Split data into {len(train_texts)} training, {len(val_texts)} validation, "
          f"and {len(test_texts)} test examples")
    
    # Create datasets
    train_dataset = ToxicityDataset(
        train_texts, train_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # If char_vocab wasn't provided, use the one created by the train_dataset
    if char_vocab is None:
        char_vocab = train_dataset.char_vocab
    
    val_dataset = ToxicityDataset(
        val_texts, val_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    test_dataset = ToxicityDataset(
        test_texts, test_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, char_vocab

def create_ood_test_set(input_path, output_path, criteria='long_texts', sample_size=500):
    print(f"Creating OOD test set with criteria: {criteria}")
    
    # Load original data
    df = pd.read_csv(input_path)
    text_column = CONFIG.get('text_column', 'comment')
    
    # Calculate text lengths
    df['text_length'] = df[text_column].apply(len)
    mean_length = df['text_length'].mean()
    std_length = df['text_length'].std()
    
    # Select examples based on criteria
    if criteria == 'long_texts':
        # Select texts that are longer than average (more than 1 std above mean)
        threshold = mean_length + std_length
        filtered_df = df[df['text_length'] > threshold]
        print(f"Selected {len(filtered_df)} texts longer than {threshold:.1f} characters")
        
    elif criteria == 'short_texts':
        # Select texts that are shorter than average (more than 1 std below mean)
        threshold = max(10, mean_length - std_length)  # Ensure minimum length
        filtered_df = df[df['text_length'] < threshold]
        print(f"Selected {len(filtered_df)} texts shorter than {threshold:.1f} characters")
        
    elif criteria == 'rare_words':
        # Get vocab frequency from the entire dataset
        import re
        from collections import Counter
        
        # Tokenize and count all words
        all_words = []
        for text in df[text_column]:
            words = re.findall(r'\b\w+\b', str(text).lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Calculate rarity score for each text (average inverse frequency of words)
        rarity_scores = []
        for text in df[text_column]:
            words = re.findall(r'\b\w+\b', str(text).lower())
            if not words:
                rarity_scores.append(0)
                continue
            
            # Calculate average rarity
            rarity = sum(1 / (word_counts[word] + 1) for word in words) / len(words)
            rarity_scores.append(rarity)
        
        df['rarity_score'] = rarity_scores
        
        # Select texts with high rarity scores
        threshold = np.percentile(rarity_scores, 75)  # Top 25% rarest
        filtered_df = df[df['rarity_score'] > threshold]
        print(f"Selected {len(filtered_df)} texts with rare vocabulary")
        
    else:  # Default to random sampling
        # Randomly sample examples
        filtered_df = df.sample(min(sample_size, len(df)), random_state=CONFIG.get('seed', 42))
        print(f"Randomly selected {len(filtered_df)} texts")
    
    # Ensure we don't have too many examples
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(sample_size, random_state=CONFIG.get('seed', 42))
    
    # Save to output path
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved OOD test set with {len(filtered_df)} examples to {output_path}")
    
    return output_path