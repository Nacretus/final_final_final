import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score, accuracy_score
from simple_model_architecture import SimplifiedCharCNNBiLSTM
from classifier_chain_model import ClassifierChainModel  # Import the new model
from CONFIG import CONFIG

def train_classifier_chain(model, train_loader, val_loader, num_epochs=30, learning_rate=0.0001):
    """
    Train the classifier chain model.
    
    Args:
        model: The classifier chain model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Trained model, best validation metrics, and metrics history
    """
    print(f"Training classifier chain model for {num_epochs} epochs with learning rate {learning_rate}")
    
    # Get device
    device = next(model.parameters()).device
    
    # Setup loss functions for binary classifiers
    toxicity_criterion = nn.BCEWithLogitsLoss()
    category_criteria = {
        'insult': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG.get('category_weights', [1.0])[0]], device=device)),
        'profanity': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG.get('category_weights', [1.0, 1.0])[1]], device=device)),
        'threat': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG.get('category_weights', [1.0, 1.0, 1.0])[2]], device=device)),
        'identity_hate': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG.get('category_weights', [1.0, 1.0, 1.0, 1.0])[3]], device=device))
    }
    severity_criterion = nn.BCEWithLogitsLoss()
    
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
            labels = batch['labels'].to(device)
            
            # Extract labels:
            # toxicity_level is labels[:, 0]
            # categories are labels[:, 1:5]
            toxicity_level = labels[:, 0].long()
            categories = labels[:, 1:5]
            
            # Convert toxicity level to necessary formats for chain:
            # 1. Binary toxicity (0 = not toxic, 1 = toxic or very toxic)
            binary_toxicity = (toxicity_level > 0).float().unsqueeze(1)
            
            # 2. Severity (0 = toxic, 1 = very toxic) - only for toxic items
            severity = (toxicity_level == 2).float().unsqueeze(1)
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio']
            ], dim=1).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(char_ids, toxicity_features)
            
            # Calculate losses
            
            # 1. Toxicity binary loss
            toxicity_binary_loss = toxicity_criterion(outputs['toxicity_binary'], binary_toxicity)
            
            # 2. Category losses - calculate for all, but weight by toxicity
            cat_insult_loss = category_criteria['insult'](
                outputs['category_logits']['insult'], 
                categories[:, 0:1]
            )
            
            cat_profanity_loss = category_criteria['profanity'](
                outputs['category_logits']['profanity'], 
                categories[:, 1:2]
            )
            
            cat_threat_loss = category_criteria['threat'](
                outputs['category_logits']['threat'], 
                categories[:, 2:3]
            )
            
            cat_identity_hate_loss = category_criteria['identity_hate'](
                outputs['category_logits']['identity_hate'], 
                categories[:, 3:4]
            )
            
            category_loss = cat_insult_loss + cat_profanity_loss + cat_threat_loss + cat_identity_hate_loss
            
            # 3. Severity loss - only meaningful for toxic items
            # Create a mask for toxic items
            toxic_mask = binary_toxicity.bool()
            
            if toxic_mask.sum() > 0:  # If there are toxic items in batch
                severity_loss = severity_criterion(
                    outputs['severity_logits'][toxic_mask],
                    severity[toxic_mask]
                )
            else:
                severity_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = toxicity_binary_loss + category_loss + severity_loss
            
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
        val_preds = []
        val_toxicity_labels = []
        val_category_preds = []
        val_category_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                char_ids = batch['char_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Extract labels
                toxicity_level = labels[:, 0].long()
                categories = labels[:, 1:5]
                
                # Convert toxicity level to necessary formats for chain
                binary_toxicity = (toxicity_level > 0).float().unsqueeze(1)
                severity = (toxicity_level == 2).float().unsqueeze(1)
                
                # Get toxicity features
                toxicity_features = torch.stack([
                    batch['all_caps_ratio'],
                    batch['toxic_keyword_count'],
                    batch['toxic_keyword_ratio']
                ], dim=1).to(device)
                
                # Forward pass
                outputs = model(char_ids, toxicity_features)
                
                # Calculate losses (same as training)
                toxicity_binary_loss = toxicity_criterion(outputs['toxicity_binary'], binary_toxicity)
                
                cat_insult_loss = category_criteria['insult'](
                    outputs['category_logits']['insult'], 
                    categories[:, 0:1]
                )
                
                cat_profanity_loss = category_criteria['profanity'](
                    outputs['category_logits']['profanity'], 
                    categories[:, 1:2]
                )
                
                cat_threat_loss = category_criteria['threat'](
                    outputs['category_logits']['threat'], 
                    categories[:, 2:3]
                )
                
                cat_identity_hate_loss = category_criteria['identity_hate'](
                    outputs['category_logits']['identity_hate'], 
                    categories[:, 3:4]
                )
                
                category_loss = cat_insult_loss + cat_profanity_loss + cat_threat_loss + cat_identity_hate_loss
                
                # Severity loss - only for toxic items
                toxic_mask = binary_toxicity.bool()
                
                if toxic_mask.sum() > 0:
                    severity_loss = severity_criterion(
                        outputs['severity_logits'][toxic_mask],
                        severity[toxic_mask]
                    )
                else:
                    severity_loss = torch.tensor(0.0, device=device)
                
                # Combined loss
                loss = toxicity_binary_loss + category_loss + severity_loss
                
                # Track loss
                val_loss += loss.item()
                
                # Make predictions
                predictions = model.predict(char_ids, toxicity_features)
                
                # Track toxicity predictions
                val_preds.extend(predictions['toxicity_level'].cpu().numpy())
                val_toxicity_labels.extend(toxicity_level.cpu().numpy())
                
                # Track category predictions
                batch_category_preds = torch.stack([
                    predictions['categories']['insult'],
                    predictions['categories']['profanity'],
                    predictions['categories']['threat'],
                    predictions['categories']['identity_hate']
                ], dim=1).cpu().numpy()
                
                val_category_preds.extend(batch_category_preds)
                val_category_labels.extend(categories.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_toxicity_acc = accuracy_score(val_toxicity_labels, val_preds)
        
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

def evaluate_classifier_chain(model, dataloader):
    """
    Evaluate the classifier chain model on a dataset.
    
    Args:
        model: The classifier chain model
        dataloader: DataLoader for evaluation data
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating classifier chain model...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Track predictions and labels
    all_toxicity_preds = []
    all_toxicity_labels = []
    all_category_preds = []
    all_category_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in dataloader:
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Extract labels
            toxicity_level = labels[:, 0].long()
            categories = labels[:, 1:5]
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio']
            ], dim=1).to(device)
            
            # Make predictions
            predictions = model.predict(char_ids, toxicity_features)
            
            # Track toxicity predictions
            all_toxicity_preds.extend(predictions['toxicity_level'].cpu().numpy())
            all_toxicity_labels.extend(toxicity_level.cpu().numpy())
            
            # Track category predictions
            batch_category_preds = torch.stack([
                predictions['categories']['insult'],
                predictions['categories']['profanity'],
                predictions['categories']['threat'],
                predictions['categories']['identity_hate']
            ], dim=1).cpu().numpy()
            
            all_category_preds.extend(batch_category_preds)
            all_category_labels.extend(categories.cpu().numpy())
    
    # Convert to arrays
    all_toxicity_preds = np.array(all_toxicity_preds)
    all_toxicity_labels = np.array(all_toxicity_labels)
    all_category_preds = np.array(all_category_preds)
    all_category_labels = np.array(all_category_labels)
    
    # Calculate metrics
    toxicity_accuracy = accuracy_score(all_toxicity_labels, all_toxicity_preds)
    
    # Calculate toxicity class metrics
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
    category_columns = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    
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
    
    # Print detailed results
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