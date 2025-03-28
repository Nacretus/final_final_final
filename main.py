from dataprocessing import *
from simple_model_architecture import *
from training_eval import *
from classifier_chain_model import ClassifierChainModel, MCDropoutChainModel
from classifier_chain_training import train_classifier_chain, evaluate_classifier_chain
from classifier_chain_integration import batch_predict_with_chain, interactive_chain_prediction

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the training and evaluation pipeline with classifier chain."""
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
        
        # Updated training parameters
        'focal_alpha': [1.5, 1.0, 1.0],            # Increased weight for non-toxic samples
        'category_weights': [1.8, 1.5, 1.8, 2.0],  # Higher weights for insult, threat and identity_hate
        'category_thresholds': [0.80, 0.85, 0.75, 0.75], # Lower thresholds for better recall
        'category_loss_scale': 1.0,                # Increased scaling for category loss
        'learning_rate': 0.0003,
        'weight_decay': 8e-4,  # L2 regularization
        'num_epochs': 30,
        'early_stopping_patience': 5,
        'use_language_detection': True,
        'use_gradient_clipping': True,
        'gradient_clip_value': 1.0,
        
        # Monte Carlo Dropout settings
        'mc_dropout_samples': 20,
        'uncertainty_threshold': 0.1,  # Threshold for high uncertainty warning
        
        # Paths
        'data_path': '17000datas.csv',
        'output_dir': 'output_chainV2',
        
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
    
    # Create base model for feature extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = SimplifiedCharCNNBiLSTM(
        n_chars=char_vocab.n_chars,
        char_emb_dim=CONFIG['char_emb_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    # Create the classifier chain model
    print("\nInitializing classifier chain model...")
    chain_model = ClassifierChainModel(base_model).to(device)
    
    # Train the classifier chain model
    print("\nTraining classifier chain model...")
    chain_model, best_metrics, metrics_history = train_classifier_chain(
        chain_model, train_loader, val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # Evaluate on test set
    print("\nEvaluating classifier chain model on test set...")
    test_metrics = evaluate_classifier_chain(chain_model, test_loader)
    
    # Create an OOD test set
    from dataprocessing import create_ood_test_set
    
    ood_data_path = os.path.join(CONFIG['output_dir'], 'ood_test_data.csv')
    create_ood_test_set(CONFIG['data_path'], ood_data_path, criteria='long_texts')
    
    # NEW: Evaluate on OOD data
    print("\nEvaluating on OOD test data...")
    ood_texts, ood_labels = load_data_from_csv(
        ood_data_path,
        text_column=CONFIG['text_column'],
        toxicity_column=CONFIG['toxicity_column'],
        category_columns=CONFIG['category_columns']
    )
    
    # Create OOD dataset and dataloader
    ood_dataset = ToxicityDataset(ood_texts, ood_labels, char_vocab)
    ood_loader = DataLoader(ood_dataset, batch_size=CONFIG['batch_size'])
    
    # Evaluate on OOD data
    print("\nEvaluating classifier chain model on OOD test set...")
    ood_metrics = evaluate_classifier_chain(chain_model, ood_loader)
    
    # Calculate performance gap between in-distribution and OOD
    id_accuracy = test_metrics['toxicity_accuracy']
    ood_accuracy = ood_metrics['toxicity_accuracy']
    gap = id_accuracy - ood_accuracy
    
    print(f"\nPerformance Gap Analysis:")
    print(f"In-distribution accuracy: {id_accuracy:.4f}")
    print(f"Out-of-distribution accuracy: {ood_accuracy:.4f}")
    print(f"Gap: {gap:.4f} ({gap/id_accuracy*100:.1f}% drop)")
    
    if gap > 0.1:  # More than 10% drop
        print("WARNING: Large performance gap detected, suggesting potential overfitting!")
    
    # NEW: Monte Carlo Dropout evaluation
    print("\nPerforming Monte Carlo Dropout evaluation on test set...")
    mc_model = MCDropoutChainModel(chain_model)
    
    # Sample texts for MC evaluation
    sample_idxs = torch.randperm(len(test_loader.dataset))[:10]  # Sample 10 examples
    sample_batch = [test_loader.dataset[idx.item()] for idx in sample_idxs]
    
    # Stack inputs
    sample_char_ids = torch.stack([item['char_ids'] for item in sample_batch]).to(device)
    sample_features = torch.stack([
        torch.tensor([item['all_caps_ratio'], item['toxic_keyword_count'], item['toxic_keyword_ratio']])
        for item in sample_batch
    ]).to(device)
    
    # Run MC evaluation
    mc_results = mc_model.predict_with_uncertainty(
        sample_char_ids, 
        sample_features,
        num_samples=CONFIG['mc_dropout_samples']
    )
    
    # Print MC results summary
    print("\nMonte Carlo Dropout Uncertainty Estimation (sample of test examples):")
    for i in range(len(sample_batch)):
        text = sample_batch[i]['text']
        toxicity_level = mc_results['toxicity_level'][i].item()
        uncertainty = mc_results['uncertainty']['overall'][i].item()
        
        toxicity_labels = ['not toxic', 'toxic', 'very toxic']
        print(f"Example {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  - Predicted: {toxicity_labels[toxicity_level]}")
        print(f"  - Uncertainty: {uncertainty:.4f}")
        
        if uncertainty > CONFIG['uncertainty_threshold']:
            print(f"  - HIGH UNCERTAINTY: prediction may be unreliable")
        print()
    
    # Save model
    model_save_path = os.path.join(CONFIG['output_dir'], 'chain_model.pth')
    torch.save(chain_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save vocabulary
    vocab_save_path = os.path.join(CONFIG['output_dir'], 'char_vocab.pkl')
    char_vocab.save(vocab_save_path)
    print(f"Vocabulary saved to {vocab_save_path}")
    
    # Run interactive prediction with updated model
    print("\nStarting interactive prediction with updated chain model...")
    interactive_chain_prediction(chain_model, char_vocab)
    
    print("\nEvaluation complete!")
    return chain_model, char_vocab, test_metrics, ood_metrics

if __name__ == "__main__":
    main()