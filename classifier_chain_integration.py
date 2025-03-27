import os
import torch
import numpy as np
from simple_model_architecture import SimplifiedCharCNNBiLSTM
from classifier_chain_model import ClassifierChainModel
from classifier_chain_training import train_classifier_chain, evaluate_classifier_chain
from dataprocessing import load_data_from_csv, create_data_loaders
from CONFIG import CONFIG

def run_classifier_chain_pipeline():
    """
    Run the complete pipeline with the classifier chain model.
    """
    print("=== Running Toxicity Detection Pipeline with Classifier Chain ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
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
    
    # Step 1: Initialize the base model for feature extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = SimplifiedCharCNNBiLSTM(
        n_chars=char_vocab.n_chars,
        char_emb_dim=CONFIG['char_emb_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    # Step 2: Create the classifier chain model using the base model
    print("Initializing classifier chain model...")
    chain_model = ClassifierChainModel(base_model).to(device)
    
    # Step 3: Train the classifier chain model
    print("\nTraining classifier chain model...")
    chain_model, best_metrics, metrics_history = train_classifier_chain(
        chain_model, train_loader, val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # Step 4: Evaluate on test set
    print("\nEvaluating classifier chain model on test set...")
    test_metrics = evaluate_classifier_chain(chain_model, test_loader)
    
    # Step 5: Save the model
    model_save_path = os.path.join(CONFIG['output_dir'], 'chain_model.pth')
    torch.save(chain_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save vocabulary
    vocab_save_path = os.path.join(CONFIG['output_dir'], 'char_vocab.pkl')
    char_vocab.save(vocab_save_path)
    print(f"Vocabulary saved to {vocab_save_path}")
    
    # Return the model and related objects
    return chain_model, char_vocab, best_metrics, test_metrics

def batch_predict_with_chain(model, texts, char_vocab, batch_size=32):
    """
    Make batch predictions using the classifier chain model.
    
    Args:
        model: The classifier chain model
        texts: List of text inputs
        char_vocab: Character vocabulary
        batch_size: Batch size for processing
    
    Returns:
        List of prediction results
    """
    results = []
    device = next(model.parameters()).device
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Preprocess and encode
        from dataprocessing import preprocess_text, extract_toxicity_features
        preprocessed_texts = [preprocess_text(text) for text in batch_texts]
        char_ids_list = [char_vocab.encode_text(text, CONFIG.get('max_chars', 300)) for text in preprocessed_texts]
        char_ids_np = np.array(char_ids_list)
        char_ids_tensor = torch.tensor(char_ids_np, dtype=torch.long).to(device)
        
        # Extract toxicity features
        toxicity_features_list = [extract_toxicity_features(text) for text in preprocessed_texts]
        toxicity_features_tensor = torch.tensor([
            [features['all_caps_ratio'], features['toxic_keyword_count'], features['toxic_keyword_ratio']]
            for features in toxicity_features_list
        ], dtype=torch.float).to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model.predict(char_ids_tensor, toxicity_features_tensor)
        
        # Process results
        for j in range(len(batch_texts)):
            # Extract predictions for this example
            toxicity_level = predictions['toxicity_level'][j].item()
            
            # Convert categories to dictionary
            category_results = {}
            for idx, category in enumerate(CONFIG['category_columns']):
                category_results[category] = bool(predictions['categories'][category][j].item())
            
            # Get probabilities
            probabilities = {}
            for category, value in predictions['probabilities'].items():
                if isinstance(value, dict):  # For category probs
                    probabilities[category] = value[j].item()
                else:
                    probabilities[category] = value[j].item()
            
            # Create result dictionary
            toxicity_levels = ['not toxic', 'toxic', 'very toxic']
            result = {
                'text': batch_texts[j],
                'toxicity': {
                    'label': toxicity_levels[toxicity_level],
                    'level': toxicity_level,
                    'probability': predictions['probabilities']['toxicity'][j].item(),
                },
                'categories': {
                    category: {
                        'detected': detected,
                        'probability': predictions['probabilities'][category][j].item()
                    }
                    for category, detected in category_results.items()
                },
                'severity': {
                    'probability': predictions['probabilities']['severity'][j].item()
                },
                'toxicity_features': toxicity_features_list[j]
            }
            
            results.append(result)
    
    return results

def interactive_chain_prediction(model, char_vocab):
    """
    Interactive prediction with the classifier chain model.
    
    Args:
        model: The classifier chain model
        char_vocab: Character vocabulary
    """
    print("\n=== Interactive Prediction with Classifier Chain Model ===")
    print("Type 'exit' to quit")
    
    while True:
        # Get text input
        text = input("\nEnter text to classify: ")
        
        if text.lower() == 'exit':
            break
        
        # Get prediction
        results = batch_predict_with_chain(model, [text], char_vocab)
        result = results[0]
        
        # Display prediction
        print("\n=== Classification Results ===")
        print(f"Text: {text}")
        print(f"\nToxicity: {result['toxicity']['label'].upper()} (Level {result['toxicity']['level']})")
        print(f"Confidence: {result['toxicity']['probability']:.4f}")
        
        if result['toxicity']['level'] > 0:  # If toxic
            print(f"Severity: {'Very Toxic' if result['toxicity']['level'] == 2 else 'Toxic'}")
            print(f"Severity Confidence: {result['severity']['probability']:.4f}")
            
            # Display detected categories
            print("\nDetected Categories:")
            detected_categories = []
            
            for category, info in result['categories'].items():
                if info['detected']:
                    detected_categories.append(category)
                    print(f"  - {category.upper()}")
                    print(f"    Confidence: {info['probability']:.4f}")
            
            if not detected_categories:
                print("  None")
        
        # Display toxicity features
        features = result['toxicity_features']
        print("\nToxicity Features:")
        print(f"  ALL CAPS Usage: {features['all_caps_ratio']:.2f} ({features['all_caps_ratio']*100:.1f}% of words)")
        print(f"  Toxic Keywords: {features['toxic_keyword_count']} ({features['toxic_keyword_ratio']*100:.1f}% of words)")
        
        if features['detected_keywords']:
            print("  Detected Keywords:")
            for keyword in features['detected_keywords']:
                print(f"    - '{keyword}'")
    
    print("\nExiting interactive prediction.")