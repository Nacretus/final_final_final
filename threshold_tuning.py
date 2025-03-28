import os
import pickle
import numpy as np
import pandas as pd
import torch
from simple_model_architecture import SimplifiedCharCNNBiLSTM
from classifier_chain_model import ClassifierChainModel
from dataprocessing import extract_toxicity_features, preprocess_text
from CONFIG import CONFIG

def load_model_and_vocab():
    """
    Load the saved model and vocabulary.
    """
    model_path = os.path.join(CONFIG['output_dir'], 'chain_model.pth')
    vocab_path = os.path.join(CONFIG['output_dir'], 'char_vocab.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Model ({model_path}) or vocabulary ({vocab_path}) not found.")
        print("Please train the model first.")
        return None, None
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        char_vocab = pickle.load(f)
    
    # Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_model = SimplifiedCharCNNBiLSTM(
        n_chars=char_vocab.n_chars,
        char_emb_dim=CONFIG['char_emb_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    chain_model = ClassifierChainModel(base_model).to(device)
    
    # Load model weights
    print(f"Loading model from {model_path}...")
    chain_model.load_state_dict(torch.load(model_path, map_location=device))
    chain_model.eval()
    
    print("Model and vocabulary loaded successfully.")
    return chain_model, char_vocab

def analyze_misclassifications(model, char_vocab, test_data_path=None):
    """
    Analyze misclassifications to help tune thresholds.
    
    Args:
        model: Trained classifier chain model
        char_vocab: Character vocabulary
        test_data_path: Path to test data CSV
    """
    from dataprocessing import load_data_from_csv
    from torch.utils.data import DataLoader
    from dataprocessing import ToxicityDataset
    
    # Use default test data if not provided
    if test_data_path is None:
        test_data_path = CONFIG['data_path']
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    texts, labels = load_data_from_csv(
        test_data_path,
        text_column=CONFIG['text_column'],
        toxicity_column=CONFIG['toxicity_column'],
        category_columns=CONFIG['category_columns']
    )
    
    # Create test dataset
    test_dataset = ToxicityDataset(texts, labels, char_vocab)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Get device
    device = next(model.parameters()).device
    model.eval()
    
    # Track predictions and labels
    all_texts = []
    all_toxicity_preds = []
    all_toxicity_probs = []
    all_toxicity_labels = []
    all_languages = []
    
    # Prediction loop
    print("Making predictions...")
    with torch.no_grad():
        for batch in test_loader:
            char_ids = batch['char_ids'].to(device)
            toxicity_labels = batch['labels'][:, 0].long()
            text_batch = batch['text']
            
            # Try to get toxicity features
            try:
                toxicity_features = torch.stack([
                    batch['all_caps_ratio'],
                    batch['toxic_keyword_count'],
                    batch['toxic_keyword_ratio']
                ], dim=1).to(device)
            except:
                # If features not available in dataset, extract them now
                toxicity_features_list = [extract_toxicity_features(text) for text in text_batch]
                toxicity_features = torch.tensor([
                    [features['all_caps_ratio'], features['toxic_keyword_count'], features['toxic_keyword_ratio']]
                    for features in toxicity_features_list
                ], dtype=torch.float).to(device)
            
            # Try to get language information if available
            languages = []
            for i, text in enumerate(text_batch):
                if hasattr(batch, 'language') and i < len(batch['language']):
                    languages.append(batch['language'][i])
                else:
                    from dataprocessing import detect_language
                    languages.append(detect_language(text))
            
            # Make predictions with default thresholds
            outputs = model.predict(char_ids, toxicity_features)
            
            # Track results
            all_texts.extend(text_batch)
            all_toxicity_preds.extend(outputs['toxicity_level'].cpu().numpy())
            all_toxicity_probs.extend(outputs['probabilities']['toxicity'].cpu().numpy())
            all_toxicity_labels.extend(toxicity_labels.numpy())
            all_languages.extend(languages)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'text': all_texts,
        'predicted': all_toxicity_preds,
        'probability': all_toxicity_probs,
        'actual': all_toxicity_labels,
        'language': all_languages
    })
    
    # Identify misclassifications
    results_df['correct'] = results_df['predicted'] == results_df['actual']
    results_df['error_type'] = 'correct'
    
    # Mark false positives and false negatives
    fp_mask = (results_df['predicted'] > 0) & (results_df['actual'] == 0)
    fn_mask = (results_df['predicted'] == 0) & (results_df['actual'] > 0)
    
    results_df.loc[fp_mask, 'error_type'] = 'false_positive'
    results_df.loc[fn_mask, 'error_type'] = 'false_negative'
    
    # Calculate statistics
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = correct / total
    
    false_positives = fp_mask.sum()
    false_negatives = fn_mask.sum()
    
    # Analyze by language
    language_stats = {}
    for lang in results_df['language'].unique():
        lang_df = results_df[results_df['language'] == lang]
        lang_total = len(lang_df)
        lang_correct = lang_df['correct'].sum()
        lang_accuracy = lang_correct / lang_total if lang_total > 0 else 0
        lang_fp = ((lang_df['predicted'] > 0) & (lang_df['actual'] == 0)).sum()
        lang_fn = ((lang_df['predicted'] == 0) & (lang_df['actual'] > 0)).sum()
        
        language_stats[lang] = {
            'total': lang_total,
            'correct': lang_correct,
            'accuracy': lang_accuracy,
            'false_positives': lang_fp,
            'false_negatives': lang_fn,
            'fp_rate': lang_fp / lang_total if lang_total > 0 else 0,
            'fn_rate': lang_fn / lang_total if lang_total > 0 else 0
        }
    
    # Print statistics
    print("\n=== Classification Statistics ===")
    print(f"Total examples: {total}")
    print(f"Correctly classified: {correct} ({accuracy:.4f})")
    print(f"False positives: {false_positives} ({false_positives/total:.4f})")
    print(f"False negatives: {false_negatives} ({false_negatives/total:.4f})")
    
    print("\n=== Language-Specific Statistics ===")
    for lang, stats in language_stats.items():
        print(f"Language: {lang}")
        print(f"  Total examples: {stats['total']}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  False positive rate: {stats['fp_rate']:.4f}")
        print(f"  False negative rate: {stats['fn_rate']:.4f}")
    
    # Analyze problematic examples
    print("\n=== False Positive Analysis ===")
    fp_df = results_df[fp_mask].copy()
    
    # Look at the most confident false positives
    most_confident_fps = fp_df.sort_values('probability', ascending=False).head(10)
    print("\nMost confident false positives:")
    for i, row in most_confident_fps.iterrows():
        print(f"Text: {row['text'][:50]}{'...' if len(row['text']) > 50 else ''}")
        print(f"Probability: {row['probability']:.4f}, Language: {row['language']}")
        print("---")
    
    # Group false positives by probability range
    fp_df['prob_range'] = pd.cut(fp_df['probability'], 
                               bins=[0, 0.7, 0.8, 0.9, 1.0],
                               labels=['0.0-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    fp_by_range = fp_df.groupby(['language', 'prob_range']).size().reset_index(name='count')
    print("\nFalse positives by probability range and language:")
    print(fp_by_range)
    
    # Save detailed results to CSV
    output_path = os.path.join(CONFIG['output_dir'], 'misclassification_analysis.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")
    
    # Recommend threshold adjustments
    print("\n=== Threshold Adjustment Recommendations ===")
    
    # Calculate optimal thresholds by language
    lang_thresholds = {}
    
    for lang, lang_group in results_df.groupby('language'):
        # For non-toxic examples, find threshold that would correctly classify most
        non_toxic = lang_group[lang_group['actual'] == 0]
        if len(non_toxic) > 0:
            # Find 95th percentile of probabilities for non-toxic examples
            threshold_95 = np.percentile(non_toxic['probability'], 95)
            
            # Find threshold that would classify 95% correctly
            threshold_sorted = sorted(non_toxic['probability'])
            idx_95 = int(len(threshold_sorted) * 0.95)
            if idx_95 < len(threshold_sorted):
                threshold_95_alt = threshold_sorted[idx_95]
            else:
                threshold_95_alt = 1.0
            
            # Use the lower of the two
            threshold = max(min(threshold_95, threshold_95_alt), 0.7)
        else:
            threshold = 0.7  # Default
        
        lang_thresholds[lang] = {
            'toxicity': threshold,
            'insult': threshold + 0.05,
            'profanity': threshold + 0.1,
            'threat': threshold,
            'identity_hate': threshold,
            'severity': 0.5
        }
    
    # Print recommended thresholds
    print("\nRecommended language-specific thresholds:")
    for lang, thresholds in lang_thresholds.items():
        print(f"'{lang}': {{")
        for key, value in thresholds.items():
            print(f"    '{key}': {value:.2f},")
        print("},")
    
    return results_df, lang_thresholds

def tune_thresholds_interactive(model, char_vocab):
    """
    Interactive tool to tune thresholds based on sample texts.
    
    Args:
        model: Trained classifier chain model
        char_vocab: Character vocabulary
    """
    from classifier_chain_integration import batch_predict_with_chain
    
    print("\n=== Interactive Threshold Tuning ===")
    print("This tool helps you find optimal thresholds for your use case.")
    print("Enter example texts and adjust thresholds until satisfied.")
    
    # Start with default thresholds
    current_thresholds = {
        'en': {
            'toxicity': 0.75,
            'insult': 0.80,
            'profanity': 0.82,
            'threat': 0.75,
            'identity_hate': 0.72,
            'severity': 0.55
        },
        'tl': {
            'toxicity': 0.87,
            'insult': 0.85,
            'profanity': 0.90,
            'threat': 0.80,
            'identity_hate': 0.80,
            'severity': 0.60
        }
    }
    
    # Store sample texts for repeated testing
    sample_texts = []
    
    while True:
        print("\nOptions:")
        print("1. Add a sample text")
        print("2. Test current thresholds on all samples")
        print("3. Adjust thresholds")
        print("4. Save thresholds to CONFIG")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Add a sample text
            text = input("\nEnter a sample text: ")
            expected = input("Is this text toxic? (y/n): ").lower() == 'y'
            
            sample_texts.append({
                'text': text,
                'expected_toxic': expected
            })
            
            print(f"Added sample (total samples: {len(sample_texts)})")
            
        elif choice == '2':
            # Test current thresholds on all samples
            if not sample_texts:
                print("No sample texts added yet. Please add some first.")
                continue
            
            print("\n=== Testing Current Thresholds ===")
            correct = 0
            
            for i, sample in enumerate(sample_texts):
                text = sample['text']
                expected_toxic = sample['expected_toxic']
                
                # Get prediction
                result = batch_predict_with_chain(model, [text], char_vocab)[0]
                
                # Check if prediction matches expectation
                predicted_toxic = result['toxicity']['level'] > 0
                is_correct = predicted_toxic == expected_toxic
                
                if is_correct:
                    correct += 1
                
                # Display result
                print(f"\nSample {i+1}: {'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"Text: {text}")
                print(f"Expected: {'Toxic' if expected_toxic else 'Not Toxic'}")
                print(f"Predicted: {'Toxic' if predicted_toxic else 'Not Toxic'} (Level {result['toxicity']['level']})")
                print(f"Probability: {result['toxicity']['probability']:.4f}")
                print(f"Language: {result['language']}")
                
                if 'applied_thresholds' in result:
                    thresholds = result['applied_thresholds']
                    print(f"Applied thresholds: toxicity={thresholds['toxicity']:.2f}, "
                          f"insult={thresholds['insult']:.2f}, "
                          f"profanity={thresholds['profanity']:.2f}")
            
            # Show overall accuracy
            accuracy = correct / len(sample_texts) if sample_texts else 0
            print(f"\nOverall accuracy: {correct}/{len(sample_texts)} ({accuracy:.2f})")
            
        elif choice == '3':
            # Adjust thresholds
            print("\n=== Current Thresholds ===")
            for lang, thresholds in current_thresholds.items():
                print(f"Language: {lang}")
                for key, value in thresholds.items():
                    print(f"  {key}: {value:.2f}")
            
            lang = input("\nWhich language to adjust? (en/tl): ").lower()
            if lang not in current_thresholds:
                print(f"Invalid language: {lang}")
                continue
            
            threshold_type = input("Which threshold to adjust? (toxicity/insult/profanity/threat/identity_hate/severity): ").lower()
            if threshold_type not in current_thresholds[lang]:
                print(f"Invalid threshold type: {threshold_type}")
                continue
            
            try:
                new_value = float(input(f"New value for {lang}.{threshold_type} (current: {current_thresholds[lang][threshold_type]:.2f}): "))
                if 0 <= new_value <= 1:
                    current_thresholds[lang][threshold_type] = new_value
                    print(f"Updated {lang}.{threshold_type} to {new_value:.2f}")
                else:
                    print("Value must be between 0 and 1")
            except ValueError:
                print("Invalid input. Please enter a number.")
            
        elif choice == '4':
            # Save thresholds to CONFIG
            import json
            
            output_path = os.path.join(CONFIG['output_dir'], 'custom_thresholds.json')
            with open(output_path, 'w') as f:
                json.dump(current_thresholds, f, indent=4)
            
            print(f"Thresholds saved to {output_path}")
            
            # Generate Python code for CONFIG
            print("\nAdd the following to your CONFIG.py:")
            print("\nLANGUAGE_THRESHOLDS = {")
            for lang, thresholds in current_thresholds.items():
                print(f"    '{lang}': {{")
                for key, value in thresholds.items():
                    print(f"        '{key}': {value:.2f},")
                print("    },")
            print("}")
            
        elif choice == '5':
            # Exit
            print("Exiting threshold tuning.")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    # Load model and vocabulary
    model, char_vocab = load_model_and_vocab()
    
    if model is None or char_vocab is None:
        print("Failed to load model or vocabulary. Exiting.")
        exit(1)
    
    print("\nSelect an option:")
    print("1. Analyze misclassifications and recommend thresholds")
    print("2. Interactive threshold tuning")
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == '1':
        analyze_misclassifications(model, char_vocab)
    elif choice == '2':
        tune_thresholds_interactive(model, char_vocab)
    else:
        print("Invalid choice.")