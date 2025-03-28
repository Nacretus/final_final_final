import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from dataprocessing import ToxicityDataset, load_data_from_csv, create_ood_test_set
from CONFIG import CONFIG

def evaluate_ood_performance(model, char_vocab, test_data_path, ood_data_path=None, criteria='long_texts'):
    """
    Comprehensive OOD evaluation comparing in-distribution and out-of-distribution performance.
    
    Args:
        model: The trained model to evaluate
        char_vocab: Character vocabulary
        test_data_path: Path to in-distribution test data
        ood_data_path: Path to existing OOD test data (optional)
        criteria: Criteria for creating OOD data if ood_data_path is None
        
    Returns:
        Dictionary with evaluation results
    """
    device = next(model.parameters()).device
    model.eval()
    print("\n=== OOD Performance Evaluation ===")
    
    # Step 1: Load in-distribution test data
    print("Loading in-distribution test data...")
    id_texts, id_labels = load_data_from_csv(
        test_data_path,
        text_column=CONFIG.get('text_column', 'comment'),
        toxicity_column=CONFIG.get('toxicity_column', 'toxicity_level'),
        category_columns=CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    )
    
    # Create in-distribution dataset and dataloader
    id_dataset = ToxicityDataset(id_texts, id_labels, char_vocab)
    id_loader = DataLoader(
        id_dataset, 
        batch_size=CONFIG.get('batch_size', 32),
        shuffle=False
    )
    
    # Step 2: Load or create OOD test data
    if ood_data_path is None:
        import os
        ood_data_path = os.path.join(CONFIG.get('output_dir', 'output'), 'ood_test_data.csv')
        create_ood_test_set(test_data_path, ood_data_path, criteria=criteria)
    
    print(f"Loading OOD test data from {ood_data_path}...")
    ood_texts, ood_labels = load_data_from_csv(
        ood_data_path,
        text_column=CONFIG.get('text_column', 'comment'),
        toxicity_column=CONFIG.get('toxicity_column', 'toxicity_level'),
        category_columns=CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    )
    
    # Create OOD dataset and dataloader
    ood_dataset = ToxicityDataset(ood_texts, ood_labels, char_vocab)
    ood_loader = DataLoader(
        ood_dataset, 
        batch_size=CONFIG.get('batch_size', 32),
        shuffle=False
    )
    
    # Step 3: Evaluate on in-distribution data
    print("\nEvaluating on in-distribution data...")
    id_results = evaluate_on_dataset(model, id_loader)
    
    # Step 4: Evaluate on OOD data
    print("\nEvaluating on out-of-distribution data...")
    ood_results = evaluate_on_dataset(model, ood_loader)
    
    # Step 5: Analyze performance gap
    print("\n=== Performance Gap Analysis ===")
    metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    
    print("Toxicity Classification:")
    for metric in metrics:
        id_value = id_results['toxicity'][metric]
        ood_value = ood_results['toxicity'][metric]
        gap = id_value - ood_value
        print(f"  {metric}: ID={id_value:.4f}, OOD={ood_value:.4f}, Gap={gap:.4f} ({gap/id_value*100:.1f}%)")
    
    print("\nCategory Classification:")
    for category in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate']):
        id_f1 = id_results['categories'][category]['f1']
        ood_f1 = ood_results['categories'][category]['f1']
        gap = id_f1 - ood_f1
        print(f"  {category}: ID F1={id_f1:.4f}, OOD F1={ood_f1:.4f}, Gap={gap:.4f} ({gap/max(0.001, id_f1)*100:.1f}%)")
    
    # Step 6: Analyze error patterns on OOD
    print("\n=== OOD Error Analysis ===")
    
    # Find samples with high confidence errors
    high_conf_errors = []
    for i, (pred, true) in enumerate(zip(ood_results['toxicity']['predictions'], ood_results['toxicity']['labels'])):
        if pred != true:
            confidence = max(ood_results['toxicity']['probabilities'][i])
            if confidence > 0.8:  # High confidence threshold
                high_conf_errors.append({
                    'text': ood_texts[i][:50] + ('...' if len(ood_texts[i]) > 50 else ''),
                    'predicted': pred,
                    'true': true,
                    'confidence': confidence
                })
    
    print(f"Found {len(high_conf_errors)} high-confidence errors in OOD data")
    if high_conf_errors:
        print("\nSample of high-confidence errors:")
        for i, error in enumerate(high_conf_errors[:5]):  # Show up to 5 examples
            print(f"Example {i+1}: {error['text']}")
            print(f"  Predicted: {error['predicted']} (conf: {error['confidence']:.4f}), True: {error['true']}")
    
    # Return combined results
    return {
        'in_distribution': id_results,
        'out_of_distribution': ood_results,
        'high_confidence_errors': high_conf_errors
    }

def evaluate_on_dataset(model, dataloader):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    toxicity_preds = []
    toxicity_labels = []
    toxicity_probs = []
    
    category_preds = {cat: [] for cat in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])}
    category_labels = {cat: [] for cat in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])}
    category_probs = {cat: [] for cat in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])}
    
    with torch.no_grad():
        for batch in dataloader:
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio']
            ], dim=1).to(device)
            
            # Make predictions
            predictions = model.predict(char_ids, toxicity_features)
            
            # Track toxicity predictions
            batch_toxicity_preds = predictions['toxicity_level'].cpu().numpy()
            batch_toxicity_labels = labels[:, 0].long().cpu().numpy()
            batch_toxicity_probs = predictions['probabilities']['toxicity'].cpu().numpy()
            
            toxicity_preds.extend(batch_toxicity_preds)
            toxicity_labels.extend(batch_toxicity_labels)
            toxicity_probs.extend(batch_toxicity_probs)
            
            # Track category predictions
            for i, category in enumerate(CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])):
                batch_category_preds = predictions['categories'][category].cpu().numpy()
                batch_category_labels = labels[:, i+1].cpu().numpy()
                batch_category_probs = predictions['probabilities'][category].cpu().numpy()
                
                category_preds[category].extend(batch_category_preds)
                category_labels[category].extend(batch_category_labels)
                category_probs[category].extend(batch_category_probs)
    
    # Calculate toxicity metrics
    toxicity_accuracy = accuracy_score(toxicity_labels, toxicity_preds)
    toxicity_conf_matrix = confusion_matrix(toxicity_labels, toxicity_preds, labels=[0, 1, 2])
    
    try:
        toxicity_report = classification_report(
            toxicity_labels, toxicity_preds,
            labels=[0, 1, 2],
            target_names=['Not Toxic', 'Toxic', 'Very Toxic'],
            output_dict=True
        )
    except Exception as e:
        print(f"Warning: Could not generate complete classification report: {e}")
        toxicity_report = {'accuracy': toxicity_accuracy}
    
    # Calculate category metrics
    category_metrics = {}
    for category in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate']):
        category_accuracy = accuracy_score(category_labels[category], category_preds[category])
        
        try:
            category_f1 = f1_score(category_labels[category], category_preds[category], zero_division=0)
            category_precision = precision_score(category_labels[category], category_preds[category], zero_division=0)
            category_recall = recall_score(category_labels[category], category_preds[category], zero_division=0)
        except:
            # Fall back if there are issues with the metrics
            category_f1 = 0.0
            category_precision = 0.0
            category_recall = 0.0
        
        category_metrics[category] = {
            'accuracy': category_accuracy,
            'f1': category_f1,
            'precision': category_precision,
            'recall': category_recall
        }
    
    # Calculate aggregate metrics
    toxicity_macro_f1 = np.mean([toxicity_report.get(c, {}).get('f1-score', 0) 
                               for c in ['Not Toxic', 'Toxic', 'Very Toxic']])
    
    toxicity_weighted_f1 = np.average(
        [toxicity_report.get(c, {}).get('f1-score', 0) for c in ['Not Toxic', 'Toxic', 'Very Toxic']],
        weights=[toxicity_report.get(c, {}).get('support', 0) for c in ['Not Toxic', 'Toxic', 'Very Toxic']]
    )
    
    category_macro_f1 = np.mean([metrics['f1'] for metrics in category_metrics.values()])
    
    return {
        'toxicity': {
            'accuracy': toxicity_accuracy,
            'conf_matrix': toxicity_conf_matrix,
            'report': toxicity_report,
            'macro_f1': toxicity_macro_f1,
            'weighted_f1': toxicity_weighted_f1,
            'predictions': toxicity_preds,
            'labels': toxicity_labels,
            'probabilities': toxicity_probs
        },
        'categories': category_metrics,
        'category_macro_f1': category_macro_f1,
        'category_predictions': category_preds,
        'category_labels': category_labels,
        'category_probabilities': category_probs
    }

from sklearn.metrics import precision_score, recall_score