import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
from dataprocessing import ToxicityDataset 
from CONFIG import*
# =============================================================================
# Conservative Parameter Adjustment Functions
# =============================================================================

def apply_conservative_adjustments(model, toxicity_criterion, category_criterion, error_analysis):
    adjustments_log = {}
    recommendations = error_analysis.get('recommendations', [])
    
    # Define maximum adjustment limits
    MAX_WEIGHT_ADJUSTMENT = 0.2  # Maximum 20% change in weights
    MIN_THRESHOLD = 0.3
    MAX_THRESHOLD = 0.8
    
    for rec in recommendations:
        if isinstance(rec, str) and rec.startswith("adjust_"):
            parts = rec.split(":")
            if len(parts) >= 2:
                adjustment_type = parts[0]
                adjustment_target = parts[1] if len(parts) > 1 else None
                adjustment_direction = parts[2] if len(parts) > 2 else None
                
                # 1. Handle class weight adjustments for toxicity - use bounded adjustments
                if adjustment_type == "adjust_class_weights" and hasattr(toxicity_criterion, 'weight'):
                    old_weights = toxicity_criterion.weight.clone().cpu()
                    
                    if adjustment_target == "decrease_tox":
                        # Use additive rather than multiplicative adjustment
                        delta = toxicity_criterion.weight[0] * MAX_WEIGHT_ADJUSTMENT
                        toxicity_criterion.weight[0] += delta  # Increase non-toxic weight
                        toxicity_criterion.weight[1:] -= delta * 0.5  # Decrease toxic weights
                        
                        adjustments_log['toxicity_weights'] = {
                            'old': old_weights.tolist(),
                            'new': toxicity_criterion.weight.cpu().tolist(),
                            'reason': 'Conservative decrease in toxic sensitivity'
                        }
                        
                    elif adjustment_target == "increase_tox":
                        # Use additive rather than multiplicative adjustment
                        delta = toxicity_criterion.weight[0] * MAX_WEIGHT_ADJUSTMENT
                        toxicity_criterion.weight[0] -= delta  # Decrease non-toxic weight
                        toxicity_criterion.weight[1:] += delta * 0.5  # Increase toxic weights
                        
                        adjustments_log['toxicity_weights'] = {
                            'old': old_weights.tolist(),
                            'new': toxicity_criterion.weight.cpu().tolist(),
                            'reason': 'Conservative increase in toxic sensitivity'
                        }
                
                # 2. Handle category threshold adjustments with bounds
                elif adjustment_type == "adjust_category_threshold":
                    category_names = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
                    if adjustment_target in category_names:
                        cat_idx = category_names.index(adjustment_target)
                        current_threshold = CONFIG['category_thresholds'][cat_idx]
                        
                        if adjustment_direction == "increase":
                            # Increase threshold with upper bound
                            new_threshold = min(MAX_THRESHOLD, current_threshold + 0.05)
                            CONFIG['category_thresholds'][cat_idx] = new_threshold
                            adjustments_log[f'{adjustment_target}_threshold'] = {
                                'old': current_threshold,
                                'new': new_threshold,
                                'reason': 'Small threshold increase'
                            }
                            
                        elif adjustment_direction == "decrease":
                            # Decrease threshold with lower bound
                            new_threshold = max(MIN_THRESHOLD, current_threshold - 0.05)
                            CONFIG['category_thresholds'][cat_idx] = new_threshold
                            adjustments_log[f'{adjustment_target}_threshold'] = {
                                'old': current_threshold,
                                'new': new_threshold,
                                'reason': 'Small threshold decrease'
                            }
    
    # No direct model bias adjustments - these are too aggressive
    return adjustments_log

# =============================================================================
# Improved Feedback System
# =============================================================================

class ImprovedFeedbackManager:
    """Redesigned feedback manager that avoids overfitting to small batches."""
    
    def __init__(self, model, char_vocab):
        self.model = model
        self.char_vocab = char_vocab
        self.feedback_examples = []
        self.original_model_state = copy.deepcopy(model.state_dict())
        self.original_thresholds = CONFIG.get('category_thresholds', [0.5, 0.5, 0.5, 0.5]).copy()
        self.min_feedback_for_retraining = 100  # Require more examples before retraining
        
    def add_feedback(self, text, model_prediction, correct_toxicity, correct_categories=None):
        """Add a single feedback example."""
        self.feedback_examples.append({
            'text': text,
            'pred_toxicity': model_prediction['toxicity']['level'],
            'true_toxicity': correct_toxicity,
            'pred_categories': [1 if model_prediction['categories'][cat]['detected'] else 0 
                               for cat in CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])],
            'true_categories': correct_categories
        })
        
        print(f"Feedback recorded. Total examples: {len(self.feedback_examples)}")
        print(f"Need {self.min_feedback_for_retraining - len(self.feedback_examples)} more examples before retraining")
        
        return len(self.feedback_examples)
    
    def perform_retraining(self, epochs=5, learning_rate=0.0001):
        """Perform retraining on accumulated feedback examples."""
        if len(self.feedback_examples) < self.min_feedback_for_retraining:
            print(f"Not enough feedback examples ({len(self.feedback_examples)}/{self.min_feedback_for_retraining})")
            return False
            
        print(f"Retraining on {len(self.feedback_examples)} feedback examples...")
        
        # Create dataset from feedback
        texts = [ex['text'] for ex in self.feedback_examples]
        labels = np.zeros((len(self.feedback_examples), 5))  # toxicity + 4 categories
        
        for i, ex in enumerate(self.feedback_examples):
            labels[i, 0] = ex['true_toxicity']
            if ex['true_categories'] is not None:
                for j, cat_val in enumerate(ex['true_categories']):
                    if j < 4:
                        labels[i, j+1] = cat_val
        
        # Split feedback data into train (80%) and validation (20%)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(texts)), test_size=0.2, stratify=labels[:, 0], random_state=42
        )
        
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        # Create datasets and dataloaders
        from torch.utils.data import DataLoader
        from dataprocessing import ToxicityDataset
        
        train_dataset = ToxicityDataset(train_texts, train_labels, self.char_vocab)
        val_dataset = ToxicityDataset(val_texts, val_labels, self.char_vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Use balanced class weights
        device = next(self.model.parameters()).device
        class_weights = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float).to(device)
        toxicity_criterion = nn.CrossEntropyLoss(weight=class_weights)
        category_criterion = nn.BCEWithLogitsLoss()
        
        # Use standard Adam optimizer with weight decay
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                char_ids = batch['char_ids'].to(device)
                toxicity_labels = batch['labels'][:, 0].long().to(device)
                category_labels = batch['labels'][:, 1:].to(device)
                
                # Get toxicity features
                toxicity_features = torch.stack([
                    batch['all_caps_ratio'],
                    batch['toxic_keyword_count'],
                    batch['toxic_keyword_ratio']
                ], dim=1).to(device)
                
                optimizer.zero_grad()
                toxicity_output, category_output = self.model(char_ids, toxicity_features)
                
                toxicity_loss = toxicity_criterion(toxicity_output, toxicity_labels)
                category_loss = category_criterion(category_output, category_labels)
                
                loss = toxicity_loss + category_loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    char_ids = batch['char_ids'].to(device)
                    toxicity_labels = batch['labels'][:, 0].long().to(device)
                    category_labels = batch['labels'][:, 1:].to(device)
                    
                    # Get toxicity features
                    toxicity_features = torch.stack([
                        batch['all_caps_ratio'],
                        batch['toxic_keyword_count'],
                        batch['toxic_keyword_ratio']
                    ], dim=1).to(device)
                    
                    toxicity_output, category_output = self.model(char_ids, toxicity_features)
                    
                    toxicity_loss = toxicity_criterion(toxicity_output, toxicity_labels)
                    category_loss = category_criterion(category_output, category_labels)
                    
                    loss = toxicity_loss + category_loss
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        print("Retraining complete!")
        return True
        
    def reset_to_original_state(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.original_model_state)
        CONFIG['category_thresholds'] = self.original_thresholds.copy()
        print("Model reset to original state")
        
    def save_feedback_data(self, save_path):
        """Save feedback data to file."""
        import pickle
        feedback_data = {
            'feedback_examples': self.feedback_examples,
            'original_thresholds': self.original_thresholds
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(feedback_data, f)
            
        print(f"Feedback data saved to {save_path}")
        
    def load_feedback_data(self, load_path):
        """Load feedback data from file."""
        import pickle
        
        try:
            with open(load_path, 'rb') as f:
                feedback_data = pickle.load(f)
                
            self.feedback_examples = feedback_data.get('feedback_examples', [])
            self.original_thresholds = feedback_data.get('original_thresholds', self.original_thresholds)
            
            print(f"Loaded {len(self.feedback_examples)} feedback examples")
            return True
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            return False

# =============================================================================
# Training Function
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.0001):
    print(f"Training model for {num_epochs} epochs with learning rate {learning_rate}")
    
    # Get device
    device = next(model.parameters()).device
    
    # Setup loss functions with balanced weights
    class_weights = torch.tensor(CONFIG.get('focal_alpha', [1.0, 2.0, 2.0]), dtype=torch.float).to(device)
    toxicity_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # For multi-label category classification
    category_weights = torch.tensor(CONFIG.get('category_weights', [1.0, 1.0, 1.0, 1.0]), dtype=torch.float).to(device)
    category_criterion = nn.BCEWithLogitsLoss(pos_weight=category_weights)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=CONFIG.get('weight_decay', 5e-4)
    )
    
    # Learning rate scheduler - reduce on plateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    patience = CONFIG.get('early_stopping_patience', 5)
    patience_counter = 0
    
    # Track metrics
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_toxicity_acc': [],
        'val_category_f1': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            char_ids = batch['char_ids'].to(device)
            toxicity_labels = batch['labels'][:, 0].long().to(device)
            category_labels = batch['labels'][:, 1:].to(device)
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio']
            ], dim=1).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with toxicity features
            toxicity_output, category_output = model(char_ids, toxicity_features)
            
            # Calculate losses
            toxicity_loss = toxicity_criterion(toxicity_output, toxicity_labels)
            category_loss = category_criterion(category_output, category_labels)
            
            # Combined loss
            loss = toxicity_loss + category_loss * CONFIG.get('category_loss_scale', 1.0)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            if CONFIG.get('use_gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=CONFIG.get('gradient_clip_value', 1.0)
                )
            
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_toxicity_preds = []
        val_toxicity_labels = []
        val_category_preds = []
        val_category_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                char_ids = batch['char_ids'].to(device)
                toxicity_labels = batch['labels'][:, 0].long().to(device)
                category_labels = batch['labels'][:, 1:].to(device)
                
                # Get toxicity features
                toxicity_features = torch.stack([
                    batch['all_caps_ratio'],
                    batch['toxic_keyword_count'],
                    batch['toxic_keyword_ratio']
                ], dim=1).to(device)
                
                # Forward pass with toxicity features
                toxicity_output, category_output = model(char_ids, toxicity_features)
                
                # Calculate losses
                toxicity_loss = toxicity_criterion(toxicity_output, toxicity_labels)
                category_loss = category_criterion(category_output, category_labels)
                
                # Combined loss
                loss = toxicity_loss + category_loss * CONFIG.get('category_loss_scale', 1.0)
                
                # Track loss
                val_loss += loss.item()
                
                # Track predictions
                toxicity_preds = torch.argmax(toxicity_output, dim=1)
                val_toxicity_preds.extend(toxicity_preds.cpu().numpy())
                val_toxicity_labels.extend(toxicity_labels.cpu().numpy())
                
                # Apply thresholds for categories
                category_thresholds = CONFIG.get('category_thresholds', [0.5, 0.5, 0.5, 0.5])
                category_probs = torch.sigmoid(category_output)
                batch_category_preds = torch.zeros_like(category_labels)
                
                for i, threshold in enumerate(category_thresholds):
                    if i < category_probs.shape[1]:
                        batch_category_preds[:, i] = (category_probs[:, i] > threshold).float()
                
                val_category_preds.extend(batch_category_preds.cpu().numpy())
                val_category_labels.extend(category_labels.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_toxicity_acc = accuracy_score(val_toxicity_labels, val_toxicity_preds)
        
        # Calculate category F1 scores
        val_category_f1 = []
        for i in range(4):  # 4 categories
            # Check if this category has any positive examples
            if np.sum(np.array(val_category_labels)[:, i]) > 0:
                cat_f1 = f1_score(
                    np.array(val_category_labels)[:, i],
                    np.array(val_category_preds)[:, i]
                )
                val_category_f1.append(cat_f1)
        
        # Calculate macro-average F1 score
        val_category_macro_f1 = np.mean(val_category_f1) if val_category_f1 else 0.0
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Track metrics
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_toxicity_acc'].append(val_toxicity_acc)
        metrics_history['val_category_f1'].append(val_category_macro_f1)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Toxicity Acc: {val_toxicity_acc:.4f}, "
              f"Val Category F1: {val_category_macro_f1:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = {
                'loss': avg_val_loss,
                'toxicity_acc': val_toxicity_acc,
                'category_f1': val_category_macro_f1,
                'epoch': epoch + 1
            }
            patience_counter = 0
            print(f"New best model found at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model state
    print(f"Loading best model from epoch {best_val_metrics['epoch']}")
    model.load_state_dict(best_model_state)
    
    return model, best_val_metrics, metrics_history

# =============================================================================
# Evaluation Functions
# =============================================================================

def comprehensive_evaluation(model, char_vocab, data_path, ood_data_path=None):
    """
    Comprehensive evaluation including out-of-distribution testing.
    
    Args:
        model: The model to evaluate
        char_vocab: Character vocabulary
        data_path: Path to in-distribution test data
        ood_data_path: Path to out-of-distribution test data (optional)
    """
    model.eval()
    results = {
        'in_distribution': {},
        'out_of_distribution': {},
        'subgroup_analysis': {}
    }
    
    # 1. Load and evaluate in-distribution data
    print("Evaluating on in-distribution data...")
    in_dist_df = pd.read_csv(data_path)
    
    # In-distribution evaluation
    results['in_distribution'] = evaluate_on_dataset(model, char_vocab, in_dist_df)
    
    # 2. If available, evaluate on out-of-distribution data
    if ood_data_path and os.path.exists(ood_data_path):
        print("\nEvaluating on out-of-distribution data...")
        ood_df = pd.read_csv(ood_data_path)
        results['out_of_distribution'] = evaluate_on_dataset(model, char_vocab, ood_df)
        
        # Calculate performance gap
        id_accuracy = results['in_distribution']['toxicity_accuracy']
        ood_accuracy = results['out_of_distribution']['toxicity_accuracy']
        gap = id_accuracy - ood_accuracy
        
        print(f"\nPerformance Gap Analysis:")
        print(f"In-distribution accuracy: {id_accuracy:.4f}")
        print(f"Out-of-distribution accuracy: {ood_accuracy:.4f}")
        print(f"Gap: {gap:.4f} ({gap/id_accuracy*100:.1f}% drop)")
        
        if gap > 0.1:  # More than 10% drop
            print("WARNING: Large performance gap detected, suggesting overfitting!")
    
    # 3. Subgroup analysis on in-distribution data
    print("\nPerforming subgroup analysis...")
    
    # 3.1 Analysis by text length
    in_dist_df['text_length'] = in_dist_df[CONFIG.get('text_column', 'text')].apply(len)
    length_bins = [(0, 50), (51, 100), (101, 200), (201, 300), (301, float('inf'))]
    
    for min_len, max_len in length_bins:
        subset = in_dist_df[(in_dist_df['text_length'] >= min_len) & (in_dist_df['text_length'] <= max_len)]
        if len(subset) > 0:
            print(f"\nEvaluating on texts with length {min_len}-{max_len} ({len(subset)} examples)")
            results['subgroup_analysis'][f'length_{min_len}_{max_len}'] = evaluate_on_dataset(
                model, char_vocab, subset, batch_size=32, verbose=False
            )
    
    # 3.2 Analysis by language (if language detection is enabled)
    if CONFIG.get('use_language_detection', False):
        # Detect languages for all texts
        from dataprocessing import detect_language
        languages = []
        for text in in_dist_df[CONFIG.get('text_column', 'text')]:
            languages.append(detect_language(text))
        in_dist_df['detected_language'] = languages
        
        for language in set(languages):
            subset = in_dist_df[in_dist_df['detected_language'] == language]
            if len(subset) > 0:
                print(f"\nEvaluating on {language} texts ({len(subset)} examples)")
                results['subgroup_analysis'][f'language_{language}'] = evaluate_on_dataset(
                    model, char_vocab, subset, batch_size=32, verbose=False
                )
    
    # 3.3 Analysis by toxicity level
    toxicity_column = CONFIG.get('toxicity_column', 'toxicity_level')
    for toxicity_level in sorted(in_dist_df[toxicity_column].unique()):
        subset = in_dist_df[in_dist_df[toxicity_column] == toxicity_level]
        if len(subset) > 0:
            level_name = ['not_toxic', 'toxic', 'very_toxic'][int(toxicity_level)]
            print(f"\nEvaluating on {level_name} texts ({len(subset)} examples)")
            results['subgroup_analysis'][f'toxicity_{level_name}'] = evaluate_on_dataset(
                model, char_vocab, subset, batch_size=32, verbose=False
            )
    
    return results

def evaluate_on_dataset(model, char_vocab, df_or_dataset, batch_size=32, verbose=True):
    from torch.utils.data import Dataset, DataLoader
    
    # Define category_columns at the function level so it's always available
    category_columns = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    
    # Define EvalDataset class first - this should already have toxicity features from your ToxicityDataset
    
    model.eval()
    device = next(model.parameters()).device
    
    # Handle different input types
    if isinstance(df_or_dataset, Dataset):
        # If df_or_dataset is a Dataset, use it directly
        eval_dataset = df_or_dataset
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    else:
        # Extract data from DataFrame
        text_column = CONFIG.get('text_column', 'text')
        toxicity_column = CONFIG.get('toxicity_column', 'toxicity_level')
        # We already defined category_columns above, so no need to redefine it here
        
        # Prepare texts and labels
        texts = df_or_dataset[text_column].tolist()
        toxicity_labels = df_or_dataset[toxicity_column].astype(int).values
        
        # Create labels array
        labels = np.zeros((len(df_or_dataset), 1 + len(category_columns)))
        labels[:, 0] = toxicity_labels
        
        # Add category values if available
        for i, col in enumerate(category_columns):
            if col in df_or_dataset.columns:
                labels[:, i+1] = df_or_dataset[col].astype(int).values
        
        # Create dataset and dataloader - use your updated ToxicityDataset which includes toxicity features
        from dataprocessing import ToxicityDataset
        eval_dataset = ToxicityDataset(texts, labels, char_vocab)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Track predictions
    all_toxicity_preds = []
    all_toxicity_labels = []
    all_category_preds = []
    all_category_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in eval_loader:
            char_ids = batch['char_ids'].to(device)
            toxicity_labels = batch['labels'][:, 0].long().to(device)
            category_labels = batch['labels'][:, 1:].to(device)
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio']
            ], dim=1).to(device)
            
            # Forward pass with toxicity features
            toxicity_output, category_output = model(char_ids, toxicity_features)
            
            # Get toxicity predictions
            toxicity_preds = torch.argmax(toxicity_output, dim=1)
            
            # Apply thresholds for categories
            category_thresholds = CONFIG.get('category_thresholds', [0.5, 0.5, 0.5, 0.5])
            category_probs = torch.sigmoid(category_output)
            batch_category_preds = torch.zeros_like(category_labels)
            
            for i, threshold in enumerate(category_thresholds):
                if i < category_probs.shape[1]:
                    batch_category_preds[:, i] = (category_probs[:, i] > threshold).float()
            
            # Store predictions and labels
            all_toxicity_preds.extend(toxicity_preds.cpu().numpy())
            all_toxicity_labels.extend(toxicity_labels.cpu().numpy())
            all_category_preds.extend(batch_category_preds.cpu().numpy())
            all_category_labels.extend(category_labels.cpu().numpy())
    
    # Rest of the evaluation function remains the same...
    # Convert to arrays
    all_toxicity_preds = np.array(all_toxicity_preds)
    all_toxicity_labels = np.array(all_toxicity_labels)
    all_category_preds = np.array(all_category_preds)
    all_category_labels = np.array(all_category_labels)
    
    # Calculate metrics
    toxicity_accuracy = accuracy_score(all_toxicity_labels, all_toxicity_preds)
    
    # Calculate toxicity class metrics
    from sklearn.metrics import classification_report
    toxicity_report = classification_report(
        all_toxicity_labels, all_toxicity_preds, 
        target_names=['Not Toxic', 'Toxic', 'Very Toxic'],
        output_dict=True
    )
    
    # Build confusion matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(all_toxicity_labels, all_toxicity_preds):
        confusion_matrix[true_label, pred_label] += 1
    
    # Calculate category metrics
    category_metrics = {}
    for i, category in enumerate(category_columns):
        if i < all_category_preds.shape[1]:
            category_report = classification_report(
                all_category_labels[:, i], all_category_preds[:, i],
                target_names=[f'Non-{category}', category],
                output_dict=True
            )
            category_metrics[category] = category_report
    
    # Calculate macro-average F1 score for categories
    category_f1_scores = []
    for i, category in enumerate(category_columns):
        if i < all_category_preds.shape[1]:
            # Check if there are any positive examples
            if np.sum(all_category_labels[:, i]) > 0:
                f1 = f1_score(all_category_labels[:, i], all_category_preds[:, i])
                category_f1_scores.append(f1)
    
    category_macro_f1 = np.mean(category_f1_scores) if category_f1_scores else 0.0
    
    # Print detailed results if verbose
    if verbose:
        print("\nToxicity Classification Report:")
        print(f"Accuracy: {toxicity_accuracy:.4f}")
        print("\nConfusion Matrix (rows=true, columns=predicted):")
        print("              | Pred Not Toxic | Pred Toxic | Pred Very Toxic |")
        print("--------------+----------------+------------+-----------------|")
        print(f"True Not Toxic | {confusion_matrix[0, 0]:14d} | {confusion_matrix[0, 1]:10d} | {confusion_matrix[0, 2]:15d} |")
        print(f"True Toxic     | {confusion_matrix[1, 0]:14d} | {confusion_matrix[1, 1]:10d} | {confusion_matrix[1, 2]:15d} |")
        print(f"True Very Toxic| {confusion_matrix[2, 0]:14d} | {confusion_matrix[2, 1]:10d} | {confusion_matrix[2, 2]:15d} |")
        
        print("\nToxicity Class Metrics:")
        for cls in ['Not Toxic', 'Toxic', 'Very Toxic']:
            print(f"  {cls}:")
            print(f"    Precision: {toxicity_report[cls]['precision']:.4f}")
            print(f"    Recall: {toxicity_report[cls]['recall']:.4f}")
            print(f"    F1-score: {toxicity_report[cls]['f1-score']:.4f}")
        
        print("\nCategory Metrics:")
        for category in category_columns:
            if category in category_metrics:
                print(f"  {category.capitalize()}:")
                print(f"    Precision: {category_metrics[category][category]['precision']:.4f}")
                print(f"    Recall: {category_metrics[category][category]['recall']:.4f}")
                print(f"    F1-score: {category_metrics[category][category]['f1-score']:.4f}")
        
        print(f"\nCategory Macro-Average F1: {category_macro_f1:.4f}")
    
    # Return metrics as dictionary
    return {
        'toxicity_accuracy': toxicity_accuracy,
        'toxicity_report': toxicity_report,
        'toxicity_confusion_matrix': confusion_matrix,
        'category_metrics': category_metrics,
        'category_macro_f1': category_macro_f1
    }

# =============================================================================
# Prediction Functions
# =============================================================================

def batch_predict_with_uncertainty(mc_model, texts, char_vocab, batch_size=32, num_samples=20):
    """
    Batch prediction with uncertainty estimation using Monte Carlo Dropout.
    """
    results = []
    device = next(mc_model.parameters()).device
    
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
        
        # Get predictions with uncertainty
        mc_results = mc_model.predict_with_uncertainty(
            char_ids_tensor, 
            toxicity_features=toxicity_features_tensor, 
            num_samples=num_samples
        )
        
        # Process each prediction
        for j in range(len(batch_texts)):
            # Get mean probabilities
            toxicity_probs = mc_results['toxicity_probs'][j]
            category_probs = mc_results['category_probs'][j]
            
            # Get uncertainty estimates
            toxicity_uncertainty = mc_results['toxicity_uncertainty'][j]
            category_uncertainty = mc_results['category_uncertainty'][j]
            predictive_entropy = mc_results['predictive_entropy'][j].item()
            
            # Make class predictions
            toxicity_pred = torch.argmax(toxicity_probs).item()
            
            # Apply thresholds for categories
            category_thresholds = CONFIG.get('category_thresholds', [0.5, 0.5, 0.5, 0.5])
            category_preds = []
            for k, threshold in enumerate(category_thresholds):
                if k < len(category_probs):
                    category_preds.append((category_probs[k] > threshold).item())
                else:
                    category_preds.append(False)
            
            # Add toxicity features to results
            features = toxicity_features_list[j]
            
            # Create result
            toxicity_levels = ['not toxic', 'toxic', 'very toxic']
            category_labels = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
            
            result = {
                'toxicity': {
                    'label': toxicity_levels[toxicity_pred],
                    'level': toxicity_pred,
                    'probabilities': {
                        level: toxicity_probs[i].item() 
                        for i, level in enumerate(toxicity_levels)
                    },
                    'uncertainty': {
                        level: toxicity_uncertainty[i].item() 
                        for i, level in enumerate(toxicity_levels)
                    }
                },
                'categories': {
                    label: {
                        "detected": bool(category_preds[i]),
                        "confidence": float(category_probs[i].item()),
                        "uncertainty": float(category_uncertainty[i].item())
                    }
                    for i, label in enumerate(category_labels) if i < len(category_preds)
                },
                'overall_uncertainty': predictive_entropy,
                'toxicity_features': {
                    'all_caps_ratio': features['all_caps_ratio'],
                    'toxic_keyword_count': features['toxic_keyword_count'],
                    'toxic_keyword_ratio': features['toxic_keyword_ratio'],
                    'detected_keywords': features['detected_keywords']
                }
            }
            
            results.append(result)
    
    return results

def interactive_prediction(model, char_vocab, feedback_manager=None):
    """
    Interactive prediction with feedback.
    
    Args:
        model: The model to use for prediction
        char_vocab: Character vocabulary
        feedback_manager: Feedback manager for collecting feedback
    """
    # Create Monte Carlo Dropout wrapper for uncertainty estimation
    from simple_model_architecture import MCDropoutModel
    mc_model = MCDropoutModel(model)
    
    # Create feedback manager if not provided
    if feedback_manager is None:
        feedback_manager = ImprovedFeedbackManager(model, char_vocab)
    
    print("\n=== Interactive Prediction with Uncertainty Estimation and Keyword Analysis ===")
    print("Type 'exit' to quit, 'stats' to see feedback statistics, 'retrain' to force retraining")
    
    while True:
        # Get text input
        text = input("\nEnter text to classify: ")
        
        if text.lower() == 'exit':
            break
        elif text.lower() == 'stats':
            print(f"\nFeedback Statistics:")
            print(f"Total feedback examples: {len(feedback_manager.feedback_examples)}")
            print(f"Minimum examples needed for retraining: {feedback_manager.min_feedback_for_retraining}")
            continue
        elif text.lower() == 'retrain':
            if len(feedback_manager.feedback_examples) > 0:
                print("\nPerforming retraining based on collected feedback...")
                feedback_manager.perform_retraining()
            else:
                print("No feedback examples available for retraining.")
            continue
        
        # Get prediction with uncertainty
        results = batch_predict_with_uncertainty(mc_model, [text], char_vocab, num_samples=20)
        result = results[0]
        
        # Display prediction with uncertainty
        print("\n=== Prediction with Uncertainty ===")
        print(f"Text: {text}")
        
        # Toxicity prediction
        toxicity_pred = result['toxicity']['label']
        toxicity_level = result['toxicity']['level']
        toxicity_prob = result['toxicity']['probabilities'][toxicity_pred]
        toxicity_uncertainty = result['toxicity']['uncertainty'][toxicity_pred]
        
        print(f"\nToxicity: {toxicity_pred.upper()} (Level {toxicity_level})")
        print(f"Confidence: {toxicity_prob:.4f}")
        print(f"Uncertainty: {toxicity_uncertainty:.4f}")
        
        # Display toxicity features
        features = result['toxicity_features']
        print("\nToxicity Features:")
        print(f"  ALL CAPS Usage: {features['all_caps_ratio']:.2f} ({features['all_caps_ratio']*100:.1f}% of words)")
        print(f"  Toxic Keywords: {features['toxic_keyword_count']} ({features['toxic_keyword_ratio']*100:.1f}% of words)")
        
        if features['detected_keywords']:
            print("  Detected Keywords:")
            for keyword in features['detected_keywords']:
                print(f"    - '{keyword}'")
        
        # Detected categories
        print("\nDetected Categories:")
        detected_categories = []
        
        for category, info in result['categories'].items():
            if info['detected']:
                detected_categories.append(category)
                print(f"  - {category.upper()}")
                print(f"    Confidence: {info['confidence']:.4f}")
                print(f"    Uncertainty: {info['uncertainty']:.4f}")
        
        if not detected_categories:
            print("  None")
        
        # Overall uncertainty
        print(f"\nOverall Uncertainty: {result['overall_uncertainty']:.4f}")
        if result['overall_uncertainty'] > 0.8:
            print("  HIGH UNCERTAINTY - prediction may be unreliable")
        
        # Probabilities for all classes
        print("\nDetailed Probabilities:")
        for level, prob in result['toxicity']['probabilities'].items():
            print(f"  {level}: {prob:.4f}")
        
        # Ask for feedback
        feedback = input("\nIs this prediction correct? (y/n): ")
        
        if feedback.lower() == 'n':
            # Get correct toxicity level
            while True:
                try:
                    correct_toxicity = int(input("Enter correct toxicity level (0=not toxic, 1=toxic, 2=very toxic): "))
                    if correct_toxicity in [0, 1, 2]:
                        break
                    else:
                        print("Please enter 0, 1, or 2")
                except ValueError:
                    print("Please enter a valid number")
            
            # Ask about categories
            get_categories = input("Do you want to provide feedback on categories? (y/n): ")
            correct_categories = None
            
            if get_categories.lower() == 'y':
                correct_categories = []
                
                category_labels = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
                for category in category_labels:
                    while True:
                        try:
                            val = int(input(f"Is '{category}' present? (0=no, 1=yes): "))
                            if val in [0, 1]:
                                correct_categories.append(val)
                                break
                            else:
                                print("Please enter 0 or 1")
                        except ValueError:
                            print("Please enter a valid number")
            
            # Provide feedback
            feedback_manager.add_feedback(
                text=text,
                model_prediction=result,
                correct_toxicity=correct_toxicity,
                correct_categories=correct_categories
            )
            
        elif feedback.lower() == 'y':
            print("Thanks for confirming the prediction!")
    
    # Save feedback data
    if len(feedback_manager.feedback_examples) > 0:
        save_path = 'feedback_data.pkl'
        feedback_manager.save_feedback_data(save_path)
        print(f"\nFeedback data saved to {save_path}")
    
    return feedback_manager

