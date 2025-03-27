import torch
import torch.nn as nn
import torch.nn.functional as F
# Import from parametrizations instead of utils directly
from torch.nn.utils.parametrizations import weight_norm
from CONFIG import*
# =============================================================================
# Custom CNN and FC Layer Implementations
# =============================================================================

class CNNLayer(nn.Module):
    def __init__(self, input_channels, large_features, small_features,
                 kernel_size, pool_size=None, batch_norm=False):
        super(CNNLayer, self).__init__()
        
        # Store channels for residual connection checking
        self.input_channels = input_channels
        self.output_channels = small_features
        
        # Primary convolution
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=large_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding
        )
        
        # Batch normalization (optional)
        self.batch_norm = nn.BatchNorm1d(large_features) if batch_norm else None
        
        # Pooling layer (optional)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size) if pool_size is not None else None
        
        # Dimension reduction layer
        self.reduce = nn.Conv1d(
            in_channels=large_features,
            out_channels=small_features,
            kernel_size=1  # 1x1 convolution
        )
        
        # Batch normalization for reduction (optional)
        self.reduce_bn = nn.BatchNorm1d(small_features) if batch_norm else None
    
    def forward(self, x):
        # Store input for potential residual connection
        residual = x if self.input_channels == self.output_channels else None
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply batch normalization if present
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply pooling if it exists
        if self.pool is not None:
            x = self.pool(x)
            # Cannot use residual connection if shape changed by pooling
            residual = None
        
        # Apply dimension reduction
        x_reduced = self.reduce(x)
        
        # Apply batch normalization for reduction if present
        if self.reduce_bn is not None:
            x_reduced = self.reduce_bn(x_reduced)
        
        # Apply ReLU to reduced features
        x_reduced = F.relu(x_reduced)
        
        # Add residual connection if possible (shapes must match)
        if residual is not None:
            x = x_reduced + residual
        else:
            x = x_reduced
        
        return x

class FCLayer(nn.Module):
    def __init__(self, input_units, output_units, batch_norm=False, dropout_rate=0.4):
        super(FCLayer, self).__init__()
        
        # Linear layer - apply weight normalization after creation
        self.fc = nn.Linear(input_units, output_units)
        self.fc = weight_norm(self.fc)  # Use updated parametrization-based weight norm
        
        # Batch normalization (optional)
        self.batch_norm = nn.BatchNorm1d(output_units) if batch_norm else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Apply linear transformation
        x = self.fc(x)
        
        # Apply batch normalization if present
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

# =============================================================================
# Simplified Model Architecture
# =============================================================================

class SimplifiedCharCNNBiLSTM(nn.Module):
    def __init__(self, n_chars, n_classes=5, char_emb_dim=50, lstm_hidden_dim=64, dropout_rate=0.4):
        super(SimplifiedCharCNNBiLSTM, self).__init__()
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(n_chars, char_emb_dim, padding_idx=0)
        
        # Build the CNN layers using the custom CNNLayer class
        self.cnn_layers = nn.ModuleList()
        input_channels = char_emb_dim
        
        # Create each CNN layer based on simplified architecture
        cnn_configs = CONFIG['cnn_configs'] if 'cnn_configs' in CONFIG else [
            {'large_features': 256, 'small_features': 64, 'kernel': 7, 'pool': 3, 'batch_norm': True},
            {'large_features': 256, 'small_features': 64, 'kernel': 7, 'pool': 3, 'batch_norm': True},
            {'large_features': 256, 'small_features': 64, 'kernel': 3, 'pool': 3, 'batch_norm': True},
        ]
        
        for layer_config in cnn_configs:
            cnn_layer = CNNLayer(
                input_channels=input_channels,
                large_features=layer_config['large_features'],
                small_features=layer_config['small_features'],
                kernel_size=layer_config['kernel'],
                pool_size=layer_config.get('pool'),
                batch_norm=layer_config.get('batch_norm', False)
            )
            
            self.cnn_layers.append(cnn_layer)
            input_channels = layer_config['small_features']
        
        # FC layer before LSTM
        self.fc = FCLayer(
            input_units=input_channels,
            output_units=256,
            batch_norm=True,
            dropout_rate=dropout_rate
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # ===== NEW CODE: Feature Processing Layers =====
        # Process toxicity features
        self.feature_fc = nn.Linear(3, 16)  # 3 features to 16 dimensions
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Combine LSTM and feature outputs
        combined_dim = lstm_hidden_dim * 2 + 16  # BiLSTM output + feature dimensions
        
        # Increased dropout for better regularization
        self.dropout = nn.Dropout(dropout_rate + 0.1)
        
        # Output layers - initialize weights first, THEN apply weight_norm
        # Create the linear layers with the new combined dimension
        self.fc_toxicity = nn.Linear(combined_dim, 3)  # 3 toxicity levels
        self.fc_category = nn.Linear(combined_dim, 4)  # 4 toxicity categories
        
        # Initialize weights before applying weight_norm
        self._init_weights()
        
        # Apply weight normalization after initialization
        from torch.nn.utils.parametrizations import weight_norm
        self.fc_toxicity = weight_norm(self.fc_toxicity)
        self.fc_category = weight_norm(self.fc_category)
    
    def _init_weights(self):
        """Initialize weights with moderated scaling."""
        # Initialize embedding with normal distribution
        nn.init.normal_(self.char_embedding.weight, mean=0, std=0.1)
        
        # Initialize convolutional layers with Kaiming
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
        # Directly initialize fc_toxicity and fc_category weights (before weight_norm is applied)
        nn.init.normal_(self.fc_toxicity.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_toxicity.bias, 0)  # Neutral bias
        
        nn.init.normal_(self.fc_category.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_category.bias, -0.2)  # Slight negative bias to reduce false positives
        
        # Initialize new feature processing layers
        nn.init.xavier_normal_(self.feature_fc.weight)
        nn.init.constant_(self.feature_fc.bias, 0)
    
    def forward(self, char_ids, toxicity_features=None):
        # Character embeddings (batch_size, seq_len, char_emb_dim)
        char_embeds = self.char_embedding(char_ids)
        
        # Convolutional layers expect (batch_size, channel, seq_len)
        x = char_embeds.permute(0, 2, 1)
        
        # Apply each CNN layer
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape for FC layer
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        
        # Apply FC layer to each position in the sequence
        x_fc = torch.zeros(batch_size, seq_len, 256, device=x.device)
        for i in range(seq_len):
            x_fc[:, i] = self.fc(x[:, i])
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(x_fc)
        lstm_out = self.dropout(lstm_out)  # Apply dropout after LSTM
        
        # Global max pooling over sequence dimension
        global_max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Process additional toxicity features if provided
        if toxicity_features is not None:
            # toxicity_features should be a tensor of shape [batch_size, 3]
            # with columns: all_caps_ratio, toxic_keyword_count, toxic_keyword_ratio
            feature_vec = self.feature_fc(toxicity_features)
            feature_vec = F.relu(feature_vec)
            feature_vec = self.feature_dropout(feature_vec)
            
            # Concatenate with LSTM output
            combined = torch.cat([global_max_pool, feature_vec], dim=1)
        else:
            # If no features provided, use only LSTM output with zero padding for feature dimension
            device = global_max_pool.device
            batch_size = global_max_pool.size(0)
            feature_padding = torch.zeros(batch_size, 16, device=device)
            combined = torch.cat([global_max_pool, feature_padding], dim=1)
        
        # Final output layers
        toxicity_output = self.fc_toxicity(combined)
        category_output = self.fc_category(combined)
        
        return toxicity_output, category_output

# =============================================================================
# Monte Carlo Dropout Wrapper
# =============================================================================

class MCDropoutModel(nn.Module):
    """Wrapper to enable Monte Carlo Dropout at inference time."""
    
    def __init__(self, base_model):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
    
    def forward(self, x, toxicity_features=None):
        return self.base_model(x, toxicity_features)
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Set dropout layers to training mode
    
    def predict_with_uncertainty(self, char_ids, toxicity_features=None, num_samples=20):
        """Run multiple forward passes with dropout enabled to estimate uncertainty."""
        self.eval()  # Set model to evaluation mode
        self.enable_dropout()  # But enable dropout
        
        toxicity_outputs = []
        category_outputs = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with dropout active
                toxicity_output, category_output = self(char_ids, toxicity_features)
                
                # Store outputs
                toxicity_outputs.append(F.softmax(toxicity_output, dim=1))
                category_outputs.append(torch.sigmoid(category_output))
        
        # Stack all samples
        toxicity_samples = torch.stack(toxicity_outputs)  # [num_samples, batch_size, num_classes]
        category_samples = torch.stack(category_outputs)  # [num_samples, batch_size, num_categories]
        
        # Mean prediction (your final prediction)
        mean_toxicity = toxicity_samples.mean(dim=0)  # [batch_size, num_classes]
        mean_category = category_samples.mean(dim=0)  # [batch_size, num_categories]
        
        # Uncertainty estimation (predictive entropy)
        toxicity_entropy = -torch.sum(mean_toxicity * torch.log(mean_toxicity + 1e-10), dim=1)
        
        # Uncertainty as standard deviation across samples
        toxicity_uncertainty = toxicity_samples.std(dim=0)  # [batch_size, num_classes]
        category_uncertainty = category_samples.std(dim=0)  # [batch_size, num_categories]
        
        return {
            'toxicity_probs': mean_toxicity,
            'category_probs': mean_category,
            'toxicity_uncertainty': toxicity_uncertainty,
            'category_uncertainty': category_uncertainty,
            'predictive_entropy': toxicity_entropy
        }