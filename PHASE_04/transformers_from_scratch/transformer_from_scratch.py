"""
Project 13: Transformers from Scratch
"Attention is all you need—literally"

This project implements a complete Transformer model from scratch, demonstrating
the revolutionary attention mechanism that changed everything in AI. You'll build
the components that power GPT, BERT, and all modern LLMs.

Learning Objectives:
1. Understand the attention mechanism mathematically
2. Implement multi-head self-attention from scratch
3. Build complete transformer architecture (encoder/decoder)
4. Train on sequence-to-sequence tasks
5. Visualize attention patterns and interpretability

Business Context:
Transformers revolutionized AI and created trillion-dollar markets:
- ChatGPT: $1.6B revenue in first year
- GitHub Copilot: Transforming software development
- Translation services: Breaking language barriers globally
- Content generation: Automating creative industries
- Search and recommendation: Powering modern platforms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    
    Since Transformers don't have inherent sequence order (unlike RNNs),
    we need to inject positional information. This uses sinusoidal patterns
    that the model can learn to interpret as position.
    
    Why it works:
    - Different frequencies for different dimensions
    - Allows model to easily learn relative positions
    - Can extrapolate to longer sequences than seen in training
    """
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate the div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: The core of the Transformer
    
    "Attention is all you need" - this mechanism allows the model to
    focus on relevant parts of the input sequence when processing each element.
    
    Key concepts:
    - Query (Q): What am I looking for?
    - Key (K): What information do I have?
    - Value (V): What is the actual information?
    - Multiple heads: Different attention patterns
    
    Mathematical formula:
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # For attention visualization
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, query: torch.Tensor, 
                                   key: torch.Tensor, 
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        This is the core attention mechanism:
        1. Compute attention scores: Q·K^T
        2. Scale by √d_k to prevent large values
        3. Apply mask (for padding or causality)
        4. Apply softmax to get attention weights
        5. Multiply by values to get attended output
        """
        d_k = query.size(-1)
        
        # Step 1 & 2: Compute scaled scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 3: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-head attention forward pass
        
        Process:
        1. Apply linear transformations to get Q, K, V
        2. Reshape for multiple heads
        3. Apply attention to each head
        4. Concatenate heads
        5. Apply final linear transformation
        """
        batch_size = query.size(0)
        
        # Step 1: Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Step 2: Reshape for multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Apply attention
        attention_output, self.attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)
        
        # Step 4: Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Step 5: Final linear transformation
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Applied to each position separately and identically.
    This adds non-linearity and allows the model to process
    the attended information.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm
    
    The residual connections help with gradient flow in deep networks.
    Layer normalization stabilizes training.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-Head Self-Attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder
    
    Stack of N encoder layers with embedding and positional encoding.
    This is the architecture used in BERT and similar models.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_length: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class SimpleSeq2SeqTransformer(nn.Module):
    """
    Simple Sequence-to-Sequence Transformer
    
    For demonstration purposes, we'll create a simple model that can
    learn basic sequence transformations like reversing sequences
    or simple translations.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048,
                 max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        
        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_layers, d_ff, max_length, dropout
        )
        
        # Simple decoder (just a classifier on top of encoder output)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, tgt_vocab_size)
        )
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode to target sequence
        output = self.decoder(encoder_output)
        
        return output

class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence learning
    
    Generates simple tasks like:
    - Sequence reversal: [1,2,3,4] -> [4,3,2,1]
    - Copy task: [1,2,3,4] -> [1,2,3,4]
    - Increment: [1,2,3,4] -> [2,3,4,5]
    """
    
    def __init__(self, num_samples: int = 10000, seq_length: int = 10, 
                 vocab_size: int = 50, task: str = 'reverse'):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.task = task
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate sequence pairs based on the task"""
        data = []
        
        for _ in range(self.num_samples):
            # Generate random input sequence (avoid 0 which we'll use for padding)
            src = torch.randint(1, self.vocab_size, (self.seq_length,))
            
            if self.task == 'reverse':
                tgt = torch.flip(src, dims=[0])
            elif self.task == 'copy':
                tgt = src.clone()
            elif self.task == 'increment':
                tgt = torch.clamp(src + 1, 1, self.vocab_size - 1)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            data.append((src, tgt))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TransformerTrainer:
    """
    Training and evaluation pipeline for Transformer models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.attention_patterns = []
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(src)
            
            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            tgt = tgt.view(-1)
            
            # Calculate loss and backpropagate
            loss = criterion(output, tgt)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Progress tracking
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                output = self.model(src)
                
                # Calculate loss
                output_flat = output.view(-1, output.size(-1))
                tgt_flat = tgt.view(-1)
                loss = criterion(output_flat, tgt_flat)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(output, dim=-1)
                correct += (predictions == tgt).sum().item()
                total += tgt.numel()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 0.001) -> Dict:
        """Complete training pipeline"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        print("Starting Transformer training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_accuracy = self.evaluate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  Val Accuracy: {val_accuracy:.4f}')
                print('-' * 50)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_accuracy': val_accuracy
        }

class TransformerAnalyzer:
    """
    Analysis and visualization tools for Transformer models
    """
    
    def __init__(self, model: nn.Module, output_dir: str = "outputs"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_attention_patterns(self, sample_input: torch.Tensor, vocab: dict):
        """
        Visualize attention patterns from the model
        
        This shows what parts of the input sequence each position
        is paying attention to - crucial for understanding model behavior.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model output and attention weights
            _ = self.model(sample_input.unsqueeze(0))
            
            # Extract attention weights from the last encoder layer
            if hasattr(self.model.encoder.layers[-1].self_attention, 'attention_weights'):
                attention_weights = self.model.encoder.layers[-1].self_attention.attention_weights
                
                # Take the first head of the first batch
                attention = attention_weights[0, 0].cpu().numpy()
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Convert token IDs to words for labeling
                tokens = [f"Token_{i}" for i in sample_input.cpu().numpy()]
                
                # Create heatmap
                sns.heatmap(attention, 
                           xticklabels=tokens, 
                           yticklabels=tokens,
                           annot=True, 
                           fmt='.2f', 
                           cmap='Blues',
                           ax=ax)
                
                ax.set_title('Attention Pattern Visualization\n(Darker = Higher Attention)', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Key Positions', fontsize=12)
                ax.set_ylabel('Query Positions', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/attention_patterns.png", dpi=300, bbox_inches='tight')
                plt.show()
    
    def visualize_positional_encoding(self, max_length: int = 100, d_model: int = 512):
        """
        Visualize positional encoding patterns
        
        Shows how the model encodes position information into the embeddings.
        """
        # Create positional encoding
        pe = PositionalEncoding(d_model, max_length)
        pos_encoding = pe.pe[:max_length, 0, :].numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot first 64 dimensions
        im1 = ax1.imshow(pos_encoding[:, :64].T, cmap='RdYlBu', aspect='auto')
        ax1.set_title('Positional Encoding\n(First 64 dimensions)', fontweight='bold')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Encoding Dimension')
        plt.colorbar(im1, ax=ax1)
        
        # Plot specific positions
        positions_to_plot = [1, 10, 20, 50]
        for pos in positions_to_plot:
            ax2.plot(pos_encoding[pos, :100], label=f'Position {pos}')
        
        ax2.set_title('Positional Encoding Patterns\n(First 100 dimensions)', fontweight='bold')
        ax2.set_xlabel('Encoding Dimension')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/positional_encoding.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_training_dynamics(self, train_losses: list, val_losses: list):
        """Analyze training dynamics and convergence"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss difference (overfitting indicator)
        loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
        ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Overfitting Indicator\n(Val Loss - Train Loss)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate schedule visualization
        ax3.plot(epochs, [0.001 * (0.95 ** epoch) for epoch in epochs], 'purple', linewidth=2)
        ax3.set_title('Simulated Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training stability (loss variance)
        window_size = 5
        if len(train_losses) >= window_size:
            train_variance = []
            for i in range(window_size, len(train_losses)):
                variance = np.var(train_losses[i-window_size:i])
                train_variance.append(variance)
            
            ax4.plot(range(window_size+1, len(train_losses)+1), train_variance, 'orange', linewidth=2)
            ax4.set_title('Training Stability\n(Loss Variance)', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Variance')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_sequence_prediction(self, model: nn.Module, test_loader: DataLoader,
                                      task: str = 'reverse', num_examples: int = 5):
        """
        Demonstrate model predictions on test sequences
        """
        model.eval()
        examples_shown = 0
        
        print("\n🔍 SEQUENCE PREDICTION EXAMPLES")
        print("=" * 60)
        
        with torch.no_grad():
            for src, tgt in test_loader:
                if examples_shown >= num_examples:
                    break
                
                # Get model predictions
                output = model(src)
                predictions = torch.argmax(output, dim=-1)
                
                # Show first few examples from batch
                batch_size = min(src.size(0), num_examples - examples_shown)
                
                for i in range(batch_size):
                    input_seq = src[i].cpu().numpy()
                    target_seq = tgt[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    
                    print(f"\nExample {examples_shown + 1} ({task} task):")
                    print(f"  Input:     {input_seq}")
                    print(f"  Target:    {target_seq}")
                    print(f"  Predicted: {pred_seq}")
                    
                    # Calculate accuracy for this sequence
                    accuracy = np.mean(target_seq == pred_seq)
                    print(f"  Accuracy:  {accuracy:.2%}")
                    
                    # Visual indicator
                    if accuracy == 1.0:
                        print("  Status: ✅ Perfect!")
                    elif accuracy >= 0.8:
                        print("  Status: ✓ Good")
                    else:
                        print("  Status: ❌ Needs improvement")
                    
                    examples_shown += 1
                
                if examples_shown >= num_examples:
                    break
    
    def model_architecture_summary(self, model: nn.Module):
        """Provide detailed model architecture summary"""
        print("\n🏗️ TRANSFORMER ARCHITECTURE SUMMARY")
        print("=" * 60)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Parameter Count:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        print(f"\n🧱 Architecture Details:")
        if hasattr(model, 'encoder'):
            print(f"   Encoder layers: {len(model.encoder.layers)}")
            print(f"   Model dimension: {model.d_model}")
            print(f"   Vocabulary sizes: {model.src_vocab_size} -> {model.tgt_vocab_size}")
            
            if len(model.encoder.layers) > 0:
                layer = model.encoder.layers[0]
                if hasattr(layer.self_attention, 'num_heads'):
                    print(f"   Attention heads: {layer.self_attention.num_heads}")
                    print(f"   Head dimension: {layer.self_attention.d_k}")
        
        print(f"\n⚡ Computational Complexity:")
        print(f"   Attention complexity: O(n²d) where n=sequence_length, d=model_dimension")
        print(f"   Memory scales quadratically with sequence length")
        print(f"   Parallelizable (unlike RNNs)")

def run_complete_transformer_analysis():
    """
    Complete Transformer analysis and demonstration
    """
    print("🤖 TRANSFORMERS FROM SCRATCH: ATTENTION IS ALL YOU NEED")
    print("=" * 80)
    print("This implementation demonstrates:")
    print("1. Multi-head self-attention mechanism")
    print("2. Positional encoding for sequence order")
    print("3. Complete encoder architecture")
    print("4. Training on sequence-to-sequence tasks")
    print("5. Attention pattern visualization and analysis")
    print("=" * 80)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Model hyperparameters
    config = {
        'src_vocab_size': 50,
        'tgt_vocab_size': 50,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'max_length': 100,
        'dropout': 0.1,
        'seq_length': 10,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001
    }
    
    print(f"\n📋 Model Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # 1. Create model
        print(f"\n🏗️  Building Transformer model...")
        model = SimpleSeq2SeqTransformer(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_length=config['max_length'],
            dropout=config['dropout']
        )
        
        # 2. Create datasets
        print(f"\n📊 Creating datasets...")
        tasks = ['reverse', 'copy', 'increment']
        
        for task in tasks:
            print(f"\n🎯 Training on {task} task...")
            
            # Create datasets
            train_dataset = SequenceDataset(
                num_samples=5000, 
                seq_length=config['seq_length'],
                vocab_size=config['src_vocab_size'],
                task=task
            )
            
            val_dataset = SequenceDataset(
                num_samples=1000,
                seq_length=config['seq_length'], 
                vocab_size=config['src_vocab_size'],
                task=task
            )
            
            test_dataset = SequenceDataset(
                num_samples=200,
                seq_length=config['seq_length'],
                vocab_size=config['src_vocab_size'], 
                task=task
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
            
            # 3. Train model
            trainer = TransformerTrainer(model, device)
            training_results = trainer.train(
                train_loader, val_loader, 
                num_epochs=config['num_epochs'],
                learning_rate=config['learning_rate']
            )
            
            print(f"\n✅ Training completed for {task} task!")
            print(f"   Final validation accuracy: {training_results['final_accuracy']:.4f}")
            
            # 4. Analyze results
            analyzer = TransformerAnalyzer(model, f"outputs/{task}_task")
            
            # Model architecture summary
            analyzer.model_architecture_summary(model)
            
            # Training dynamics analysis
            analyzer.analyze_training_dynamics(
                training_results['train_losses'],
                training_results['val_losses']
            )
            
            # Demonstrate predictions
            analyzer.demonstrate_sequence_prediction(
                model, test_loader, task=task, num_examples=5
            )
            
            # Visualize attention patterns (using a sample)
            sample_input = next(iter(test_loader))[0][0]  # First sequence from first batch
            analyzer.visualize_attention_patterns(sample_input, {})
            
            # Positional encoding visualization
            analyzer.visualize_positional_encoding()
            
            # Save model and results
            torch.save(model.state_dict(), f"outputs/{task}_task/transformer_model.pth")
            
            with open(f"outputs/{task}_task/training_results.json", 'w') as f:
                json.dump(training_results, f, indent=2)
            
            print(f"\n💾 Results saved to outputs/{task}_task/")
        
        # 5. Business impact analysis
        print(f"\n💼 BUSINESS IMPACT ANALYSIS")
        print("=" * 60)
        
        business_applications = [
            {
                'domain': 'Language Translation',
                'description': 'Real-time translation services',
                'market_size': '$56B by 2027',
                'key_players': 'Google Translate, DeepL, Microsoft Translator',
                'transformer_advantage': 'Better context understanding, faster inference'
            },
            {
                'domain': 'Content Generation',
                'description': 'Automated content creation',
                'market_size': '$1.2B by 2025',
                'key_players': 'OpenAI GPT, Jasper, Copy.ai',
                'transformer_advantage': 'Human-like text generation, contextual coherence'
            },
            {
                'domain': 'Code Generation',
                'description': 'AI-powered programming assistance',
                'market_size': '$2.9B by 2025',
                'key_players': 'GitHub Copilot, CodeT5, Codex',
                'transformer_advantage': 'Understanding code context, multi-language support'
            },
            {
                'domain': 'Search & Recommendation',
                'description': 'Semantic search and personalization',
                'market_size': '$15B by 2024',
                'key_players': 'Google BERT, Microsoft Turing, Amazon Alexa',
                'transformer_advantage': 'Better understanding of user intent and context'
            }
        ]
        
        for app in business_applications:
            print(f"\n📂 {app['domain']}:")
            print(f"   Description: {app['description']}")
            print(f"   Market Size: {app['market_size']}")
            print(f"   Key Players: {app['key_players']}")
            print(f"   Transformer Advantage: {app['transformer_advantage']}")
        
        print(f"\n🚀 IMPLEMENTATION ROADMAP:")
        print("Phase 1: Understanding (Weeks 1-2)")
        print("   • Study attention mechanism mathematics")
        print("   • Implement basic transformer components")
        print("   • Run toy experiments with sequence tasks")
        
        print("\nPhase 2: Application (Weeks 3-4)")
        print("   • Apply to real business problems")
        print("   • Fine-tune pre-trained models")
        print("   • Integrate with existing systems")
        
        print("\nPhase 3: Optimization (Weeks 5-6)")
        print("   • Optimize for production deployment")
        print("   • Implement caching and batching")
        print("   • Monitor performance and accuracy")
        
        print(f"\n✅ TRANSFORMER ANALYSIS COMPLETE!")
        print(f"📁 All results saved to outputs/ directory")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_transformer_analysis()
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNING OUTCOMES:")
    print("1. Attention mechanism is the key to understanding modern NLP")
    print("2. Transformers enable parallel processing unlike RNNs")
    print("3. Positional encoding is crucial for sequence understanding")
    print("4. Multi-head attention captures different types of relationships")
    print("5. Layer normalization and residual connections enable deep networks")
    print("\n🔄 NEXT STEPS:")
    print("• Experiment with different attention patterns")
    print("• Try on real-world NLP tasks")
    print("• Explore pre-trained transformer models")
    print("• Implement decoder-only models (GPT-style)")
    print("• Study recent transformer variants (RoBERTa, T5, etc.)")
    print("="*80)