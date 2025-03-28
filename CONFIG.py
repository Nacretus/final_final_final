"""
Configuration settings for toxicity detection model.
This file centralizes all parameters used throughout the application.
"""

# =============================================================================
# Data Processing Settings
# =============================================================================
DATA_SETTINGS = {
    'text_column': 'comment',              # Column name for text data
    'toxicity_column': 'toxicity_level',   # Column name for toxicity level
    'category_columns': ['insult', 'profanity', 'threat', 'identity_hate'],  # Category column names
    'max_chars': 300,                      # Maximum sequence length
    'use_language_detection': True,        # Whether to detect language
}

# =============================================================================
# Vocabulary Settings
# =============================================================================
VOCAB_SETTINGS = {
    'use_hybrid_vocabulary': True,         # Whether to use hybrid vocabulary approach
    'alphabet': "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",  # Fixed alphabet
    'max_vocab_size': 500,                 # Maximum vocabulary size
    'min_char_count': 2,                   # Minimum character count to include in vocabulary
}

# =============================================================================
# Model Architecture Settings
# =============================================================================
MODEL_SETTINGS = {
    'char_emb_dim': 50,                    # Character embedding dimension
    'lstm_hidden_dim': 64,                 # BiLSTM hidden dimension
    'dropout_rate': 0.35,                  # Dropout rate (increased)

    # CNN configurations
    'cnn_configs': [
        {'large_features': 256, 'small_features': 64, 'kernel': 7, 'pool': 3, 'batch_norm': True},
        {'large_features': 256, 'small_features': 64, 'kernel': 7, 'pool': 3, 'batch_norm': True},
        {'large_features': 256, 'small_features': 64, 'kernel': 3, 'pool': 3, 'batch_norm': True},
    ]
}

# =============================================================================
# Training Settings
# =============================================================================
TRAINING_SETTINGS = {
    'batch_size': 32,                          # Batch size
    'learning_rate': 0.0003,                   # Learning rate
    'weight_decay': 8e-4,                      # L2 regularization strength (increased)
    'num_epochs': 30,                          # Maximum training epochs
    'early_stopping_patience': 5,              # Patience for early stopping
    
    # UPDATED: Changed class weights to give more importance to non-toxic class (level 0)
    'focal_alpha': [1.5, 1.0, 1.0],            # Increased weight for non-toxic samples
    
    # UPDATED: Increased weights for harder categories
    'category_weights': [1.8, 1.5, 1.8, 2.0],  # Higher weights for insult, threat and identity_hate
    
    # UPDATED: Reduced thresholds to improve detection sensitivity
    'category_thresholds': [0.80, 0.85, 0.75, 0.75],  # Lower thresholds for better recall
    
    'category_loss_scale': 1.0,                # Increased scaling for category loss
    'use_gradient_clipping': True,             # Whether to use gradient clipping
    'gradient_clip_value': 1.0,                # Maximum gradient norm
    'num_workers': 4,                          # Number of data loading workers
    'seed': 42,                                # Random seed for reproducibility
}

# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_SETTINGS = {
    'mc_dropout_samples': 20,                  # Number of Monte Carlo samples for uncertainty
    'uncertainty_threshold': 0.1,             # Threshold for high uncertainty warning
}
# =============================================================================
# Feedback System Settings
# =============================================================================
FEEDBACK_SETTINGS = {
    # UPDATED: Reduced number of examples needed for retraining
    'min_feedback_for_retraining': 15,     # Minimum feedback examples before retraining
    'feedback_retrain_epochs': 10,         # Number of epochs for feedback retraining (increased)
    'feedback_learning_rate': 0.0001,      # Learning rate for feedback retraining
    'feedback_batch_size': 16,             # Batch size for feedback retraining
}
# =============================================================================
# Language-Specific Thresholds
# =============================================================================
LANGUAGE_THRESHOLDS = {
    'en': {
        'toxicity': 0.75,                  # Higher threshold for English
        'insult': 0.80,                    # Higher threshold for insult in English
        'profanity': 0.82,                 # Higher threshold for profanity in English
        'threat': 0.75,                    # Higher threshold for threat in English
        'identity_hate': 0.72,             # Higher threshold for identity hate in English
        'severity': 0.55                   # Severity threshold
    },
    'tl': {
        'toxicity': 0.87,                  # Much higher threshold for Tagalog to reduce false positives
        'insult': 0.85,                    # Higher threshold for insult in Tagalog
        'profanity': 0.90,                 # Much higher threshold for profanity in Tagalog
        'threat': 0.80,                    # Higher threshold for threat in Tagalog
        'identity_hate': 0.80,             # Higher threshold for identity hate in Tagalog
        'severity': 0.60                   # Severity threshold
    }
}
# =============================================================================
# Safe Word Confidence Adjustment
# =============================================================================
SAFE_WORD_SETTINGS = {
    'enable_safe_word_features': True,
    'safe_word_threshold_boost': 0.15,  # How much to increase threshold when safe words are present
    'max_threshold': 0.95,              # Maximum threshold after adjustments
    'benign_phrases': []                # Will be populated from CSV
}

# =============================================================================
# Paths
# =============================================================================
PATHS = {
    'data_path': '17000datas.csv',         # Path to main data file
    'safe_words_path':'safeword,phrases,mixed.csv',
    'output_dir': 'output_chainV2',                # Directory for saving outputs
    'model_save_path': 'output_chainV2/model.pth', # Path for saving trained model
    'vocab_save_path': 'output_chainV2/char_vocab.pkl',  # Path for saving vocabulary
    'feedback_save_path': 'output_chainV2/feedback_data.pkl',  # Path for saving feedback data
}

# =============================================================================
# Combined CONFIG dictionary (for backwards compatibility)
# =============================================================================
CONFIG = {
    # Combine all settings into one dictionary
    **DATA_SETTINGS,
    **VOCAB_SETTINGS,
    **MODEL_SETTINGS,
    **TRAINING_SETTINGS,
    **EVAL_SETTINGS,
    **FEEDBACK_SETTINGS,
    **LANGUAGE_THRESHOLDS,
    **SAFE_WORD_SETTINGS,
    **PATHS
}
import os
import pandas as pd

# Load safe words from CSV if the file exists
safe_words_path = 'safeword,phrases,mixed.csv'
if os.path.exists(safe_words_path):
    try:
        # Try different ways to read the CSV depending on its structure
        try:
            # If the CSV has headers
            safe_words_df = pd.read_csv(safe_words_path)
            # Assuming the first column contains the safe words/phrases
            column_name = safe_words_df.columns[0]  # Get the first column name
            SAFE_WORD_SETTINGS['benign_phrases'] = safe_words_df[column_name].tolist()
        except:
            # If the CSV is just a list of words with no header
            safe_words_df = pd.read_csv(safe_words_path, header=None)
            SAFE_WORD_SETTINGS['benign_phrases'] = safe_words_df[0].tolist()
        
        # Remove any NaN values and convert to lowercase
        SAFE_WORD_SETTINGS['benign_phrases'] = [
            str(phrase).lower() for phrase in SAFE_WORD_SETTINGS['benign_phrases'] 
            if str(phrase) != 'nan'
        ]
        
        print(f"Loaded {len(SAFE_WORD_SETTINGS['benign_phrases'])} safe words/phrases from {safe_words_path}")
    except Exception as e:
        print(f"Error loading safe words from CSV: {e}")
        # Keep the default list as fallback
else:
    print(f"Safe words file {safe_words_path} not found. Using default safe words.")

# Make sure the settings are also available in the main CONFIG dictionary
CONFIG.update({
    'safe_word_settings': SAFE_WORD_SETTINGS,
})

# For easy import in other modules
if __name__ == "__main__":
    print("Configuration settings loaded successfully.")
    print(f"Total parameters: {len(CONFIG)}")