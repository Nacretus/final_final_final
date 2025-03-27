from dataprocessing import*
from simple_model_architecture import*
from training_eval import*

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the training and evaluation pipeline."""
    # Define configuration
    global CONFIG
    CONFIG = {
        # Data settings
        'text_column': 'comment',
        'toxicity_column': 'toxicity_level',
        'category_columns': ['insult', 'profanity', 'threat', 'identity_hate'],
        'max_chars': 300,
        'batch_size': 32,
        'num_workers': 4,
        
        # Vocabulary settings
        'use_hybrid_vocabulary': True,
        'alphabet': "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
        'max_vocab_size': 500,
        
        # Model parameters
        'char_emb_dim': 50,
        'lstm_hidden_dim': 64,
        'dropout_rate': 0.4,
        
        # Training parameters
        'focal_alpha': [1.0, 2.0, 2.0],  # Class weights for toxicity
        'category_weights': [1.0, 1.0, 1.0, 1.0],  # Weights for categories
        'category_thresholds': [0.5, 0.5, 0.5, 0.5],  # Thresholds for categories
        'category_loss_scale': 1.0,  # Scale factor for category loss
        'learning_rate': 0.0005,
        'weight_decay': 5e-4,  # L2 regularization
        'num_epochs': 30,
        'early_stopping_patience': 5,
        'use_language_detection': True,
        'use_gradient_clipping': True,
        'gradient_clip_value': 1.0,
        
        # Paths
        'data_path': '17000datas.csv',
        'output_dir': 'output',
        
        # Seed for reproducibility
        'seed': 42
    }
    
    # Set random seed
    import torch
    import numpy as np
    import random
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
    from dataprocessing import load_data_from_csv, create_data_loaders
    
    print("Loading data...")
    texts, labels = load_data_from_csv(
        CONFIG['data_path'],
        text_column=CONFIG['text_column'],
        toxicity_column=CONFIG['toxicity_column'],
        category_columns=CONFIG['category_columns']
    )
    
    # Create data loaders and vocabulary
    train_loader, val_loader, test_loader, char_vocab = create_data_loaders(
        texts, labels,
        batch_size=CONFIG['batch_size'],
        max_len=CONFIG['max_chars'],
        detect_lang=CONFIG['use_language_detection'],
        num_workers=CONFIG['num_workers'],
        seed=CONFIG['seed']
    )
    
    # Create model
    from simple_model_architecture import SimplifiedCharCNNBiLSTM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplifiedCharCNNBiLSTM(
        n_chars=char_vocab.n_chars,
        char_emb_dim=CONFIG['char_emb_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    # Train model
    print("\nTraining model...")
    model, best_metrics, metrics_history = train_model(
        model, train_loader, val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # Save model
    model_save_path = os.path.join(CONFIG['output_dir'], 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    
    # Save vocabulary
    vocab_save_path = os.path.join(CONFIG['output_dir'], 'char_vocab.pkl')
    char_vocab.save(vocab_save_path)
    
    # Save config
    import json
    config_save_path = os.path.join(CONFIG['output_dir'], 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_on_dataset(model, char_vocab, test_loader.dataset)
    
    # Create an OOD test set
    from dataprocessing import create_ood_test_set
    
    ood_data_path = os.path.join(CONFIG['output_dir'], 'ood_test_data.csv')
    create_ood_test_set(CONFIG['data_path'], ood_data_path, criteria='long_texts')
    
    # Comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    eval_results = comprehensive_evaluation(model, char_vocab, CONFIG['data_path'], ood_data_path)
    
    # Interactive prediction
    feedback_manager = ImprovedFeedbackManager(model, char_vocab)
    interactive_prediction(model, char_vocab, feedback_manager)
    
    print("\nEvaluation complete!")
    return model, char_vocab, feedback_manager

if __name__ == "__main__":
    main()