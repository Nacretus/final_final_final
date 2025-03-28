import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class ClassifierChainModel(nn.Module):
    """
    A model implementing classifier chains for toxicity detection.
    Chain flow:
    1. Base Network -> Is text toxic? (yes/no)
    2. Base Network + Toxicity -> Which categories apply? (insult, profanity, threat, identity hate)
    3. Base Network + Toxicity + Categories -> What's the severity? (toxic/very toxic)
    """
    def __init__(self, base_model):
        """
        Initialize the classifier chain using an existing base model.
        Args:
            base_model: Existing model that extracts features from text
        """
        super(ClassifierChainModel, self).__init__()
        self.base_model = base_model
        
        # Get dimension of the base model's output
        # This should be the combined dimension from the base network
        # which includes BiLSTM output + feature dimensions
        if hasattr(base_model, 'fc_toxicity'):
            combined_dim = base_model.fc_toxicity.in_features
        else:
            # Default if not available - need to fix this!
            # The error is happening because the base_model feature processing changed
            # from 16 to 32 dimensions, but this wasn't updated here
            combined_dim = 160  # BiLSTM (64*2) + feature dimension (32)
        
        # Chain link 1: Binary toxicity classifier (toxic or not)
        self.toxicity_binary = nn.Linear(combined_dim, 1)
        # Initialize weights before applying weight norm to avoid the weight_orig issue
        nn.init.normal_(self.toxicity_binary.weight, mean=0, std=0.01)
        nn.init.constant_(self.toxicity_binary.bias, -0.8)  # Negative bias to reduce false positives
        # Apply weight norm after initialization
        self.toxicity_binary = weight_norm(self.toxicity_binary)
        
        # Chain link 2: Category classifiers (4 binary classifiers that use toxicity result)
        # Input: base features + toxicity binary result
        self.category_insult = nn.Linear(combined_dim + 1, 1)
        nn.init.normal_(self.category_insult.weight, mean=0, std=0.01)
        nn.init.constant_(self.category_insult.bias, -1.0)
        self.category_insult = weight_norm(self.category_insult)
        
        self.category_profanity = nn.Linear(combined_dim + 1, 1)
        nn.init.normal_(self.category_profanity.weight, mean=0, std=0.01)
        nn.init.constant_(self.category_profanity.bias, -1.0)
        self.category_profanity = weight_norm(self.category_profanity)
        
        self.category_threat = nn.Linear(combined_dim + 1, 1)
        nn.init.normal_(self.category_threat.weight, mean=0, std=0.01)
        nn.init.constant_(self.category_threat.bias, -1.0)
        self.category_threat = weight_norm(self.category_threat)
        
        self.category_identity_hate = nn.Linear(combined_dim + 1, 1)
        nn.init.normal_(self.category_identity_hate.weight, mean=0, std=0.01)
        nn.init.constant_(self.category_identity_hate.bias, -1.0)
        self.category_identity_hate = weight_norm(self.category_identity_hate)
        
        # Chain link 3: Severity classifier (toxic vs very toxic)
        # Input: base features + toxicity binary + all 4 categories
        self.severity = nn.Linear(combined_dim + 1 + 4, 1)
        nn.init.normal_(self.severity.weight, mean=0, std=0.01)
        nn.init.constant_(self.severity.bias, 0)
        self.severity = weight_norm(self.severity)
    
    def forward(self, char_ids, toxicity_features=None):
        """
        Forward pass implementing the classifier chain.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor
            
        Returns:
            Dictionary with all outputs from the chain
        """
        # Extract features from base model
        # Note: This implementation assumes your base model has these specific internals
        # You may need to adjust this based on your actual model implementation
        
        # Character embeddings
        char_embeds = self.base_model.char_embedding(char_ids)
        
        # Process through CNN layers
        x = char_embeds.permute(0, 2, 1)  # (batch_size, channels, seq_len)
        for cnn_layer in self.base_model.cnn_layers:
            x = cnn_layer(x)
        
        # Process through FC and LSTM
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        
        # Apply FC layer to each position
        x_fc = torch.zeros(batch_size, seq_len, 256, device=x.device)
        for i in range(seq_len):
            x_fc[:, i] = self.base_model.fc(x[:, i])
        
        # Apply BiLSTM
        lstm_out, _ = self.base_model.lstm(x_fc)
        lstm_out = self.base_model.dropout(lstm_out)
        
        # Global max pooling
        global_max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Process additional toxicity features if provided
        if toxicity_features is not None:
            # Make sure to use the feature_fc layer with appropriate dimensions
            feature_vec = self.base_model.feature_fc(toxicity_features)
            feature_vec = self.base_model.feature_bn(feature_vec)  # Use batch norm
            feature_vec = F.relu(feature_vec)
            feature_vec = self.base_model.feature_dropout(feature_vec)
            
            # Concatenate with LSTM output
            base_features = torch.cat([global_max_pool, feature_vec], dim=1)
        else:
            # If no features provided, use only LSTM output with zero padding
            device = global_max_pool.device
            batch_size = global_max_pool.size(0)
            # Use 32 dimensions here instead of 16 to match with the base model
            feature_padding = torch.zeros(batch_size, 32, device=device)
            base_features = torch.cat([global_max_pool, feature_padding], dim=1)
        
        # ===== CLASSIFIER CHAIN IMPLEMENTATION =====
        
        # Chain link 1: Binary toxicity classification (is it toxic?)
        toxicity_bin_logits = self.toxicity_binary(base_features)
        toxicity_bin_probs = torch.sigmoid(toxicity_bin_logits)
        
        # For the chain, we use the probability as a feature for the next classifiers
        # Chain link 2: Category classification with toxicity information
        # Concatenate base features with toxicity binary prediction
        features_with_toxicity = torch.cat([base_features, toxicity_bin_probs], dim=1)
        
        # Category classifiers
        insult_logits = self.category_insult(features_with_toxicity)
        profanity_logits = self.category_profanity(features_with_toxicity)
        threat_logits = self.category_threat(features_with_toxicity)
        identity_hate_logits = self.category_identity_hate(features_with_toxicity)
        
        # Get probabilities
        insult_probs = torch.sigmoid(insult_logits)
        profanity_probs = torch.sigmoid(profanity_logits)
        threat_probs = torch.sigmoid(threat_logits)
        identity_hate_probs = torch.sigmoid(identity_hate_logits)
        
        # Combine category probabilities for next chain link
        category_probs = torch.cat([
            insult_probs, profanity_probs, threat_probs, identity_hate_probs
        ], dim=1)
        
        # Chain link 3: Severity classification (toxic vs very toxic)
        # This only applies if the content is toxic
        # Concatenate base features with toxicity and all categories
        features_for_severity = torch.cat([base_features, toxicity_bin_probs, category_probs], dim=1)
        
        severity_logits = self.severity(features_for_severity)
        severity_probs = torch.sigmoid(severity_logits)
        
        # Return all outputs from the chain
        return {
            'toxicity_binary': toxicity_bin_logits,  # Is it toxic at all? (binary)
            'toxicity_binary_probs': toxicity_bin_probs,
            'category_logits': {
                'insult': insult_logits,
                'profanity': profanity_logits,
                'threat': threat_logits,
                'identity_hate': identity_hate_logits
            },
            'category_probs': {
                'insult': insult_probs,
                'profanity': profanity_probs,
                'threat': threat_probs,
                'identity_hate': identity_hate_probs
            },
            'severity_logits': severity_logits,  # How severe is the toxicity?
            'severity_probs': severity_probs
        }
    
    def adjust_thresholds_for_safe_words(self, thresholds, toxicity_features):
        """
        Adjust classification thresholds based on safe word detection.
        
        Args:
            thresholds: Dictionary of current thresholds
            toxicity_features: Features extracted from text including safe word info
            
        Returns:
            Updated thresholds dictionary
        """
        try:
            from CONFIG import SAFE_WORD_SETTINGS
            
            # Only proceed if safe word feature is enabled
            if not SAFE_WORD_SETTINGS.get('enable_safe_word_features', False):
                return thresholds
            
            # Get threshold boost amount and maximum threshold
            boost_amount = SAFE_WORD_SETTINGS.get('safe_word_threshold_boost', 0.15)
            max_threshold = SAFE_WORD_SETTINGS.get('max_threshold', 0.95)
            
            # Create a copy of thresholds to modify
            adjusted_thresholds = thresholds.copy()
            
            # Get batch size
            batch_size = len(toxicity_features) if isinstance(toxicity_features, list) else toxicity_features.size(0)
            
            # Process each example in the batch
            for i in range(batch_size):
                # Check if we have safe word information
                if isinstance(toxicity_features, list) and 'safe_word_count' in toxicity_features[i]:
                    safe_word_count = toxicity_features[i]['safe_word_count']
                elif hasattr(toxicity_features, 'safe_word_count'):
                    safe_word_count = toxicity_features.safe_word_count[i]
                else:
                    # Skip if we don't have safe word info
                    continue
                    
                # Apply threshold adjustments if safe words were detected
                if safe_word_count > 0:
                    # Scale boost based on number of safe words, up to a maximum
                    scaled_boost = min(boost_amount * safe_word_count, boost_amount * 3)
                    
                    # Adjust toxicity threshold
                    current = adjusted_thresholds['toxicity']
                    adjusted_thresholds['toxicity'] = min(max_threshold, current + scaled_boost)
                    
                    # Adjust category thresholds
                    for category in ['insult', 'profanity', 'threat', 'identity_hate']:
                        current = adjusted_thresholds[category]
                        adjusted_thresholds[category] = min(max_threshold, current + scaled_boost)
                    
            return adjusted_thresholds
        
        except Exception as e:
            # If anything goes wrong, return original thresholds
            print(f"Error adjusting thresholds for safe words: {e}")
            return thresholds
    
    def predict(self, char_ids, toxicity_features=None, thresholds=None):
        """
        Make predictions using the classifier chain with threshold application.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor
            thresholds: Dictionary of thresholds for each classifier (optional)
            
        Returns:
            Dictionary with final predictions
        """
        # Updated thresholds to reduce false positives
        if thresholds is None:
            thresholds = {
                'toxicity': 0.7,        # Increased from 0.5
                'insult': 0.6,          # Increased
                'profanity': 0.7,       # Increased to reduce false positives
                'threat': 0.6,          # Increased
                'identity_hate': 0.6,   # Increased
                'severity': 0.5         # Kept the same
            }
        
        # Adjust thresholds based on safe words if applicable
        if toxicity_features is not None:
            thresholds = self.adjust_thresholds_for_safe_words(thresholds, toxicity_features)
        
        # Get raw outputs
        outputs = self.forward(char_ids, toxicity_features)
        
        # Apply thresholds for final predictions
        is_toxic = (outputs['toxicity_binary_probs'] > thresholds['toxicity']).float()
        
        # Only predict categories if content is toxic
        # But calculate anyway for consistency
        insult = (outputs['category_probs']['insult'] > thresholds['insult']).float()
        profanity = (outputs['category_probs']['profanity'] > thresholds['profanity']).float()
        threat = (outputs['category_probs']['threat'] > thresholds['threat']).float()
        identity_hate = (outputs['category_probs']['identity_hate'] > thresholds['identity_hate']).float()
        
        # Enforce consistency: if not toxic, no categories should be positive
        insult = insult * is_toxic
        profanity = profanity * is_toxic
        threat = threat * is_toxic
        identity_hate = identity_hate * is_toxic
        
        # Determine severity
        # If not toxic at all, severity is 0
        # If toxic but severity below threshold, it's level 1 (toxic)
        # If toxic and severity above threshold, it's level 2 (very toxic)
        severity = (outputs['severity_probs'] > thresholds['severity']).float()
        
        # Determine final toxicity level
        # 0 = not toxic, 1 = toxic, 2 = very toxic
        toxicity_level = torch.zeros_like(is_toxic, dtype=torch.long)
        toxicity_level[is_toxic.squeeze() == 1] = 1  # Toxic
        toxicity_level[torch.logical_and(is_toxic.squeeze() == 1, severity.squeeze() == 1)] = 2  # Very toxic
        
        return {
            'toxicity_level': toxicity_level,
            'categories': {
                'insult': insult,
                'profanity': profanity,
                'threat': threat,
                'identity_hate': identity_hate
            },
            'probabilities': {
                'toxicity': outputs['toxicity_binary_probs'],
                'insult': outputs['category_probs']['insult'],
                'profanity': outputs['category_probs']['profanity'],
                'threat': outputs['category_probs']['threat'],
                'identity_hate': outputs['category_probs']['identity_hate'],
                'severity': outputs['severity_probs']
            }
        }


class MCDropoutChainModel(nn.Module):
    """
    Wrapper for ClassifierChainModel to enable Monte Carlo Dropout at inference time.
    Used for uncertainty estimation through multiple forward passes.
    """
    def __init__(self, chain_model):
        super(MCDropoutChainModel, self).__init__()
        self.chain_model = chain_model
    
    def forward(self, char_ids, toxicity_features=None):
        return self.chain_model(char_ids, toxicity_features)
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        # Enable dropout in the base model
        for module in self.chain_model.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
        # Also enable dropout in the classifier chain if present
        for module in self.chain_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, char_ids, toxicity_features=None, num_samples=20, thresholds=None):
        """
        Run multiple forward passes with dropout enabled to estimate uncertainty.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor  
            num_samples: Number of Monte Carlo samples
            thresholds: Dictionary of thresholds for classification
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.eval()  # Set model to evaluation mode
        self.enable_dropout()  # But enable dropout
        
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'toxicity': 0.7,
                'insult': 0.6,
                'profanity': 0.7,
                'threat': 0.6,
                'identity_hate': 0.6,
                'severity': 0.5
            }
        
        # Adjust thresholds for safe words if applicable
        if toxicity_features is not None:
            thresholds = self.chain_model.adjust_thresholds_for_safe_words(thresholds, toxicity_features)
        
        # Storage for samples
        toxicity_probs_samples = []
        insult_probs_samples = []
        profanity_probs_samples = []
        threat_probs_samples = []
        identity_hate_probs_samples = []
        severity_probs_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with dropout active
                outputs = self.chain_model(char_ids, toxicity_features)
                
                # Store probability samples
                toxicity_probs_samples.append(outputs['toxicity_binary_probs'])
                insult_probs_samples.append(outputs['category_probs']['insult'])
                profanity_probs_samples.append(outputs['category_probs']['profanity'])
                threat_probs_samples.append(outputs['category_probs']['threat'])
                identity_hate_probs_samples.append(outputs['category_probs']['identity_hate'])
                severity_probs_samples.append(outputs['severity_probs'])
        
        # Stack all samples
        toxicity_probs_samples = torch.stack(toxicity_probs_samples)  # [num_samples, batch_size, 1]
        insult_probs_samples = torch.stack(insult_probs_samples)
        profanity_probs_samples = torch.stack(profanity_probs_samples)
        threat_probs_samples = torch.stack(threat_probs_samples)
        identity_hate_probs_samples = torch.stack(identity_hate_probs_samples)
        severity_probs_samples = torch.stack(severity_probs_samples)
        
        # Mean predictions
        mean_toxicity_probs = toxicity_probs_samples.mean(dim=0)
        mean_insult_probs = insult_probs_samples.mean(dim=0)
        mean_profanity_probs = profanity_probs_samples.mean(dim=0)
        mean_threat_probs = threat_probs_samples.mean(dim=0)
        mean_identity_hate_probs = identity_hate_probs_samples.mean(dim=0)
        mean_severity_probs = severity_probs_samples.mean(dim=0)
        
        # Standard deviation (uncertainty)
        toxicity_uncertainty = toxicity_probs_samples.std(dim=0)
        insult_uncertainty = insult_probs_samples.std(dim=0)
        profanity_uncertainty = profanity_probs_samples.std(dim=0)
        threat_uncertainty = threat_probs_samples.std(dim=0)
        identity_hate_uncertainty = identity_hate_probs_samples.std(dim=0)
        severity_uncertainty = severity_probs_samples.std(dim=0)
        
        # Predictive entropy for overall uncertainty
        toxicity_entropy = -mean_toxicity_probs * torch.log(mean_toxicity_probs + 1e-10) - \
                           (1 - mean_toxicity_probs) * torch.log(1 - mean_toxicity_probs + 1e-10)
        
        # Apply thresholds to mean predictions
        is_toxic = (mean_toxicity_probs > thresholds['toxicity']).float()
        
        insult = (mean_insult_probs > thresholds['insult']).float()
        profanity = (mean_profanity_probs > thresholds['profanity']).float()
        threat = (mean_threat_probs > thresholds['threat']).float()
        identity_hate = (mean_identity_hate_probs > thresholds['identity_hate']).float()
        
        # Enforce consistency: if not toxic, no categories
        insult = insult * is_toxic
        profanity = profanity * is_toxic
        threat = threat * is_toxic
        identity_hate = identity_hate * is_toxic
        
        # Determine severity
        severity = (mean_severity_probs > thresholds['severity']).float()
        
        # Determine final toxicity level
        toxicity_level = torch.zeros_like(is_toxic, dtype=torch.long)
        toxicity_level[is_toxic.squeeze() == 1] = 1  # Toxic
        toxicity_level[torch.logical_and(is_toxic.squeeze() == 1, severity.squeeze() == 1)] = 2  # Very toxic
        
        return {
            'toxicity_level': toxicity_level,
            'categories': {
                'insult': insult,
                'profanity': profanity,
                'threat': threat,
                'identity_hate': identity_hate
            },
            'probabilities': {
                'toxicity': mean_toxicity_probs,
                'insult': mean_insult_probs,
                'profanity': mean_profanity_probs, 
                'threat': mean_threat_probs,
                'identity_hate': mean_identity_hate_probs,
                'severity': mean_severity_probs
            },
            'uncertainty': {
                'toxicity': toxicity_uncertainty,
                'insult': insult_uncertainty,
                'profanity': profanity_uncertainty,
                'threat': threat_uncertainty,
                'identity_hate': identity_hate_uncertainty,
                'severity': severity_uncertainty,
                'overall': toxicity_entropy
            }
        }