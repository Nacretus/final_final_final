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
            # Default if not available
            combined_dim = 144  # Typical value from your architecture
        
        # Chain link 1: Binary toxicity classifier (toxic or not)
        self.toxicity_binary = nn.Linear(combined_dim, 1)
        self.toxicity_binary = weight_norm(self.toxicity_binary)
        
        # Chain link 2: Category classifiers (4 binary classifiers that use toxicity result)
        # Input: base features + toxicity binary result
        self.category_insult = nn.Linear(combined_dim + 1, 1)
        self.category_profanity = nn.Linear(combined_dim + 1, 1)
        self.category_threat = nn.Linear(combined_dim + 1, 1)
        self.category_identity_hate = nn.Linear(combined_dim + 1, 1)
        
        # Apply weight normalization to category classifiers
        self.category_insult = weight_norm(self.category_insult)
        self.category_profanity = weight_norm(self.category_profanity)
        self.category_threat = weight_norm(self.category_threat)
        self.category_identity_hate = weight_norm(self.category_identity_hate)
        
        # Chain link 3: Severity classifier (toxic vs very toxic)
        # Input: base features + toxicity binary + all 4 categories
        self.severity = nn.Linear(combined_dim + 1 + 4, 1)
        self.severity = weight_norm(self.severity)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with suitable values for classification."""
        for m in [self.toxicity_binary, self.category_insult, self.category_profanity,
                self.category_threat, self.category_identity_hate, self.severity]:
            # Handle both older and newer PyTorch weight_norm implementations
            if hasattr(m, 'weight_orig'):
                # Older PyTorch versions
                nn.init.normal_(m.weight_orig, mean=0, std=0.01)
            elif hasattr(m, 'weight'):
                # Newer PyTorch versions (using parametrizations)
                nn.init.normal_(m.weight, mean=0, std=0.01)
            
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, char_ids, toxicity_features=None):
        """
        Forward pass implementing the classifier chain.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor
            
        Returns:
            Dictionary with all outputs from the chain
        """
        # Get base features from the original model's feature extraction
        # We'll use the implementation details of your base_model to extract features
        
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
            feature_vec = self.base_model.feature_fc(toxicity_features)
            feature_vec = F.relu(feature_vec)
            feature_vec = self.base_model.feature_dropout(feature_vec)
            
            # Concatenate with LSTM output
            base_features = torch.cat([global_max_pool, feature_vec], dim=1)
        else:
            # If no features provided, use only LSTM output with zero padding
            device = global_max_pool.device
            batch_size = global_max_pool.size(0)
            feature_padding = torch.zeros(batch_size, 16, device=device)
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
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'toxicity': 0.5,
                'insult': 0.5,
                'profanity': 0.5,
                'threat': 0.5,
                'identity_hate': 0.5,
                'severity': 0.5
            }
        
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