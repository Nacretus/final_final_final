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
    'dropout_rate': 0.45,                  # Dropout rate (increased)

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
    'batch_size': 32,                      # Batch size
    'learning_rate': 0.0005,               # Learning rate
    'weight_decay': 5e-4,                  # L2 regularization strength (increased)
    'num_epochs': 30,                      # Maximum training epochs
    'early_stopping_patience': 5,          # Patience for early stopping
    # UPDATED: Changed class weights to give more importance to non-toxic class (level 0)
    'focal_alpha': [1.0, 2.0, 2.0],        # Class weights for toxicity levels
    # UPDATED: Increased weights for insult category to improve detection
    'category_weights': [1.0, 1.0, 1.0, 1.0],  # Weights for categories
    # UPDATED: Increased thresholds to reduce false positives
    'category_thresholds': [0.5, 0.5, 0.5, 0.5],  # Thresholds for binary categories
    'category_loss_scale': 1.2,            # Scaling factor for category loss (increased)
    'use_gradient_clipping': True,         # Whether to use gradient clipping
    'gradient_clip_value': 1.0,            # Maximum gradient norm
    'num_workers': 4,                      # Number of data loading workers
    'seed': 42,                            # Random seed for reproducibility
}

# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_SETTINGS = {
    'mc_dropout_samples': 20,              # Number of Monte Carlo samples for uncertainty
    'uncertainty_threshold': 0.8,          # Threshold for high uncertainty warning
}

# =============================================================================
# Feedback System Settings
# =============================================================================
FEEDBACK_SETTINGS = {
    # UPDATED: Reduced number of examples needed for retraining
    'min_feedback_for_retraining': 20,     # Minimum feedback examples before retraining
    'feedback_retrain_epochs': 10,         # Number of epochs for feedback retraining (increased)
    'feedback_learning_rate': 0.0001,      # Learning rate for feedback retraining
    'feedback_batch_size': 16,             # Batch size for feedback retraining
}

# =============================================================================
# Paths
# =============================================================================
PATHS = {
    'data_path': '17000datas.csv',         # Path to main data file
    'output_dir': 'output_chains',                # Directory for saving outputs
    'model_save_path': 'output_chains/model.pth', # Path for saving trained model
    'vocab_save_path': 'output_chains/char_vocab.pkl',  # Path for saving vocabulary
    'feedback_save_path': 'output_chains/feedback_data.pkl',  # Path for saving feedback data
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
    **PATHS
}

# For easy import in other modules
if __name__ == "__main__":
    print("Configuration settings loaded successfully.")
    print(f"Total parameters: {len(CONFIG)}")