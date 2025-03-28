import os
import torch
import numpy as np
from simple_model_architecture import SimplifiedCharCNNBiLSTM
from classifier_chain_model import ClassifierChainModel, MCDropoutChainModel
from classifier_chain_training import train_classifier_chain, evaluate_classifier_chain
from dataprocessing import load_data_from_csv, create_data_loaders, detect_language
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

def batch_predict_with_chain(model, texts, char_vocab, batch_size=32, use_mc_dropout=False, num_mc_samples=20):
    """
    Make batch predictions using the classifier chain model.
    
    Args:
        model: The classifier chain model
        texts: List of text inputs
        char_vocab: Character vocabulary
        batch_size: Batch size for processing
        use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
        num_mc_samples: Number of MC samples if use_mc_dropout is True
    
    Returns:
        List of prediction results
    """
    results = []
    device = next(model.parameters()).device
    
    # Create MC model wrapper if needed
    if use_mc_dropout:
        from classifier_chain_model import MCDropoutChainModel
        mc_model = MCDropoutChainModel(model)
    
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
        
        # NEW: Detect language for each text for language-specific thresholds
        languages = [detect_language(text) for text in preprocessed_texts]
        
        # Get predictions
        batch_results = []
        
        for j, language in enumerate(languages):
            # Create single example tensors
            single_char_ids = char_ids_tensor[j:j+1]
            single_features = toxicity_features_tensor[j:j+1]
            
            # Use language-specific thresholds with adjusted values
            if language == 'tl':  # Tagalog
                thresholds = {
                    'toxicity': 0.65,       # Lowered threshold for Tagalog
                    'insult': 0.45,         # Lowered all thresholds
                    'profanity': 0.55,
                    'threat': 0.45,
                    'identity_hate': 0.45,
                    'severity': 0.5
                }
            else:  # Default (English)
                thresholds = {
                    'toxicity': 0.6,        # Lowered from 0.7
                    'insult': 0.4,          # Lowered from 0.6
                    'profanity': 0.5,       # Lowered from 0.7
                    'threat': 0.4,          # Lowered from 0.6
                    'identity_hate': 0.4,   # Lowered from 0.6
                    'severity': 0.5         # Kept the same
                }
            
            # Apply additional rules for specific keywords
            text = preprocessed_texts[j].lower()
            features = toxicity_features_list[j]
            
            # Adjust thresholds based on content
            if features['toxic_keyword_count'] > 0:
                # If toxic keywords present, lower the threshold more aggressively
                thresholds['toxicity'] = max(0.4, thresholds['toxicity'] - 0.15)
            else:
                # Words that often get false positives for profanity - increase threshold
                benign_words = ['love', 'things', 'adjust', 'good', 'afternoon', 'morning', 'hello']
                if any(word in text.split() for word in benign_words):
                    thresholds['toxicity'] = min(0.85, thresholds['toxicity'] + 0.1)
                    thresholds['profanity'] = min(0.85, thresholds['profanity'] + 0.1)
            
            # Get predictions with appropriate method
            with torch.no_grad():
                if use_mc_dropout:
                    predictions = mc_model.predict_with_uncertainty(
                        single_char_ids, 
                        single_features, 
                        num_samples=num_mc_samples,
                        thresholds=thresholds
                    )
                    
                    # Extract results
                    prediction_result = {
                        'toxicity_level': predictions['toxicity_level'][0].item(),
                        'categories': {
                            cat: bool(predictions['categories'][cat][0].item())
                            for cat in CONFIG['category_columns']
                        },
                        'probabilities': {
                            'toxicity': predictions['probabilities']['toxicity'][0].item(),
                            'severity': predictions['probabilities']['severity'][0].item()
                        },
                        'uncertainty': {
                            'overall': predictions['uncertainty']['overall'][0].item(),
                            'toxicity': predictions['uncertainty']['toxicity'][0].item()
                        }
                    }
                    
                    # Add category probabilities and uncertainties
                    for cat in CONFIG['category_columns']:
                        prediction_result['probabilities'][cat] = predictions['probabilities'][cat][0].item()
                        prediction_result['uncertainty'][cat] = predictions['uncertainty'][cat][0].item()
                    
                    # FIX: Use get() with default to handle missing 'severity' key
                    if 'severity' in predictions['uncertainty']:
                        prediction_result['uncertainty']['severity'] = predictions['uncertainty']['severity'][0].item()
                    else:
                        # Use overall uncertainty as a fallback for severity uncertainty
                        prediction_result['uncertainty']['severity'] = prediction_result['uncertainty']['overall']
                    
                else:
                    predictions = model.predict(single_char_ids, single_features, thresholds=thresholds)
                    
                    # Extract predictions for this example
                    prediction_result = {
                        'toxicity_level': predictions['toxicity_level'][0].item(),
                        'categories': {
                            cat: bool(predictions['categories'][cat][0].item())
                            for cat in CONFIG['category_columns']
                        },
                        'probabilities': {
                            'toxicity': predictions['probabilities']['toxicity'][0].item(),
                            'severity': predictions['probabilities']['severity'][0].item()
                        }
                    }
                    
                    # Add category probabilities
                    for cat in CONFIG['category_columns']:
                        prediction_result['probabilities'][cat] = predictions['probabilities'][cat][0].item()
            
            # Create full result dictionary
            toxicity_levels = ['not toxic', 'toxic', 'very toxic']
            result = {
                'text': batch_texts[j],
                'language': language,
                'toxicity': {
                    'label': toxicity_levels[prediction_result['toxicity_level']],
                    'level': prediction_result['toxicity_level'],
                    'probability': prediction_result['probabilities']['toxicity'],
                },
                'categories': {
                    category: {
                        'detected': prediction_result['categories'][category],
                        'probability': prediction_result['probabilities'][category]
                    }
                    for category in CONFIG['category_columns']
                },
                'severity': {
                    'probability': prediction_result['probabilities']['severity']
                },
                'toxicity_features': toxicity_features_list[j]
            }
            
            # Add uncertainty if available
            if use_mc_dropout:
                result['uncertainty'] = {
                    'overall': prediction_result['uncertainty']['overall'],
                    'toxicity': prediction_result['uncertainty']['toxicity'],
                    'categories': {
                        category: prediction_result['uncertainty'][category]
                        for category in CONFIG['category_columns']
                    },
                    'severity': prediction_result['uncertainty']['severity']
                }
            
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results

def interactive_chain_prediction(model, char_vocab, use_mc_dropout=False):
    """
    Interactive prediction with the classifier chain model.
    
    Args:
        model: The classifier chain model
        char_vocab: Character vocabulary
        use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty
    """
    if use_mc_dropout:
        print("\n=== Interactive Prediction with Classifier Chain Model (with Uncertainty) ===")
    else:
        print("\n=== Interactive Prediction with Classifier Chain Model ===")
    
    print("Type 'exit' to quit, 'mc on' to enable uncertainty estimation, 'mc off' to disable it")
    
    # Start with MC dropout off
    current_mc_state = use_mc_dropout
    
    while True:
        # Get text input
        text = input("\nEnter text to classify: ")
        
        if text.lower() == 'exit':
            break
        elif text.lower() == 'mc on':
            current_mc_state = True
            print("Monte Carlo dropout enabled - uncertainty estimation active")
            continue
        elif text.lower() == 'mc off':
            current_mc_state = False
            print("Monte Carlo dropout disabled")
            continue
        
        # Get prediction
        results = batch_predict_with_chain(
            model, [text], char_vocab, 
            use_mc_dropout=current_mc_state,
            num_mc_samples=CONFIG.get('mc_dropout_samples', 20)
        )
        result = results[0]
        
        # Display prediction
        print("\n=== Classification Results ===")
        print(f"Text: {text}")
        print(f"Detected Language: {result['language']}")
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
                    
                    # Show uncertainty if available
                    if current_mc_state and 'uncertainty' in result:
                        uncertainty = result['uncertainty']['categories'][category]
                        print(f"    Uncertainty: {uncertainty:.4f}")
                        
                        if uncertainty > CONFIG.get('uncertainty_threshold', 0.08):
                            print(f"    HIGH UNCERTAINTY - detection may be unreliable")
            
            if not detected_categories:
                print("  None")
        
        # Display uncertainty if available
        if current_mc_state and 'uncertainty' in result:
            print(f"\nOverall Uncertainty: {result['uncertainty']['overall']:.4f}")
            if result['uncertainty']['overall'] > CONFIG.get('uncertainty_threshold', 0.08):
                print("  HIGH UNCERTAINTY - prediction may be unreliable")
        
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

def batch_predict_with_uncertainty(model, texts, char_vocab, batch_size=32, num_samples=20):
    """
    Convenience wrapper for batch_predict_with_chain with MC dropout enabled.
    
    Args:
        model: The classifier chain model
        texts: List of text inputs
        char_vocab: Character vocabulary
        batch_size: Batch size for processing
        num_samples: Number of MC samples
        
    Returns:
        List of prediction results with uncertainty estimates
    """
    # Use updated thresholds
    thresholds = {
        'toxicity': 0.6,        # Lowered from 0.7
        'insult': 0.4,          # Lowered from 0.6
        'profanity': 0.5,       # Lowered from 0.7
        'threat': 0.4,          # Lowered from 0.6
        'identity_hate': 0.4,   # Lowered from 0.6
        'severity': 0.5         # Kept the same
    }
    
    try:
        results = batch_predict_with_chain(
            model, texts, char_vocab, 
            batch_size=batch_size, 
            use_mc_dropout=True, 
            num_mc_samples=num_samples
        )
        return results
    except Exception as e:
        print(f"Error in batch_predict_with_uncertainty: {e}")
        # Fallback to non-MC prediction if MC fails
        print("Falling back to standard prediction without uncertainty estimation")
        return batch_predict_with_chain(
            model, texts, char_vocab,
            batch_size=batch_size,
            use_mc_dropout=False
        )