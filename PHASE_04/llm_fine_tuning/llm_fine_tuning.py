"""
Project 16: Fine-Tuning LLMs with LoRA/PEFT
"When fine-tuning beats RAG and how to do it efficiently"

This project demonstrates how to fine-tune large language models efficiently
using Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT).
You'll learn when to choose fine-tuning over RAG and how to implement it
cost-effectively.

Learning Objectives:
1. Understand when fine-tuning is superior to RAG
2. Master LoRA and PEFT techniques for efficient training
3. Implement domain-specific model adaptation
4. Compare fine-tuning vs RAG performance
5. Deploy fine-tuned models in production

Business Context:
Fine-tuning is ideal for:
- Domain-specific language patterns (legal, medical, financial)
- Style adaptation (brand voice, writing style)
- Task-specific optimization (classification, summarization)
- Proprietary knowledge that can't be shared via RAG
- Real-time applications requiring minimal latency

When to choose Fine-tuning over RAG:
- Need consistent style/tone across all outputs
- Working with structured/domain-specific language
- Require ultra-low latency responses
- Have sufficient high-quality training data
- Need model to internalize patterns, not just facts
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning experiments"""
    model_name: str = "microsoft/DialoGPT-small"  # Start with smaller model for demo
    task_type: str = "CAUSAL_LM"  # or "SEQ_2_SEQ_LM", "SEQ_CLS"
    
    # LoRA configuration
    lora_r: int = 16              # Rank of adaptation
    lora_alpha: int = 32          # LoRA scaling parameter
    lora_dropout: float = 0.05    # Dropout probability
    target_modules: List[str] = None
    
    # Training configuration
    learning_rate: float = 3e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_length: int = 512
    
    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False
    
    # Output
    output_dir: str = "outputs"

@dataclass
class TrainingMetrics:
    """Metrics tracking during training"""
    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None

class DatasetCreator:
    """
    Create training datasets for different fine-tuning tasks
    
    Supported tasks:
    1. Text generation (Q&A, chat, creative writing)
    2. Text classification (sentiment, topic, intent)
    3. Summarization
    4. Code generation
    """
    
    def __init__(self):
        self.tokenizer = None
    
    def create_qa_dataset(self, domain: str = "customer_support") -> Dataset:
        """
        Create Q&A dataset for domain-specific fine-tuning
        
        In production, you would:
        - Collect real customer queries and responses
        - Use domain experts to create high-quality examples
        - Augment data using techniques like paraphrasing
        - Ensure diversity in question types and complexity
        """
        
        if domain == "customer_support":
            qa_pairs = [
                {
                    "question": "How do I reset my password?",
                    "answer": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and we'll send you a reset link. Follow the instructions in the email to create a new password. Make sure your new password is at least 8 characters long and includes both letters and numbers."
                },
                {
                    "question": "What is your return policy?",
                    "answer": "We offer a 30-day return policy for most items. Products must be in original condition with tags attached. To initiate a return, log into your account, go to 'Order History', and select 'Return Item'. Some items like personalized products or perishables cannot be returned. Return shipping is free for defective items."
                },
                {
                    "question": "How long does shipping take?",
                    "answer": "Standard shipping typically takes 5-7 business days. Express shipping takes 2-3 business days. Orders placed before 2 PM EST ship the same day. You'll receive a tracking number via email once your order ships. Shipping times may vary during holidays or due to weather conditions."
                },
                {
                    "question": "Do you offer international shipping?",
                    "answer": "Yes, we ship to over 50 countries worldwide. International shipping costs vary by destination and package weight. Delivery times range from 7-21 business days depending on location. Please note that customers are responsible for any customs duties or taxes. Some products may have shipping restrictions to certain countries."
                },
                {
                    "question": "How can I track my order?",
                    "answer": "You can track your order in several ways: 1) Log into your account and check 'Order History' 2) Use the tracking number sent to your email 3) Text 'TRACK' to our SMS service 4) Call our customer service line. Tracking information is usually available within 24 hours of shipment."
                },
                {
                    "question": "What payment methods do you accept?",
                    "answer": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and bank transfers. For large orders over $1000, we also accept wire transfers. All transactions are secured with SSL encryption for your protection."
                },
                {
                    "question": "Can I change or cancel my order?",
                    "answer": "You can modify or cancel orders within 1 hour of placement if they haven't shipped yet. Go to your account dashboard and select 'Manage Orders'. If your order has already shipped, you'll need to process a return instead. Rush orders typically cannot be modified once confirmed."
                },
                {
                    "question": "Do you price match?",
                    "answer": "Yes, we offer price matching for identical products sold by authorized dealers. The price must be publicly available and in stock. We don't price match auction sites, membership clubs, or clearance sales. Submit a price match request through our customer service with a link to the competitor's listing."
                },
                {
                    "question": "What warranty do you provide?",
                    "answer": "Most products come with a manufacturer's warranty ranging from 1-3 years. We also offer extended warranty options at checkout. Warranty covers manufacturing defects but not damage from misuse or normal wear. To claim warranty service, contact us with your order number and photos of the issue."
                },
                {
                    "question": "How do I contact customer service?",
                    "answer": "You can reach our customer service team through: Email (response within 24 hours), Live chat (9 AM - 8 PM EST), Phone (1-800-SUPPORT), or our mobile app. For technical issues, use our support ticket system for faster resolution. We also have extensive FAQs and video tutorials on our website."
                }
            ]
            
        elif domain == "financial_advisor":
            qa_pairs = [
                {
                    "question": "Should I invest in stocks or bonds?",
                    "answer": "The allocation between stocks and bonds depends on your risk tolerance, time horizon, and financial goals. Generally, stocks offer higher potential returns but with more volatility, while bonds provide stability and income. A common rule of thumb is to subtract your age from 100 to determine your stock percentage (e.g., a 30-year-old might consider 70% stocks, 30% bonds). However, individual circumstances vary significantly."
                },
                {
                    "question": "How much should I save for retirement?",
                    "answer": "Financial experts typically recommend saving 10-15% of your gross income for retirement. The exact amount depends on when you start saving, your desired lifestyle in retirement, and expected Social Security benefits. Starting early allows compound interest to work in your favor. If you're behind, you may need to save 20% or more. Consider maximizing employer 401(k) matching first, as it's essentially free money."
                },
                {
                    "question": "What's the best way to build an emergency fund?",
                    "answer": "An emergency fund should cover 3-6 months of living expenses, stored in a readily accessible, low-risk account like a high-yield savings account or money market fund. Start by saving $1000 as a starter emergency fund, then gradually build to the full amount. Automate transfers to make saving easier. Keep this fund separate from other savings to avoid temptation to spend it on non-emergencies."
                }
            ]
        
        else:
            # Generic dataset
            qa_pairs = [
                {
                    "question": "What is machine learning?",
                    "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that learning."
                },
                {
                    "question": "Explain deep learning in simple terms",
                    "answer": "Deep learning uses artificial neural networks with multiple layers to process data, similar to how the human brain works. Each layer learns increasingly complex features from the data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition."
                }
            ]
        
        # Format for training
        formatted_data = []
        for pair in qa_pairs:
            # Create instruction-following format
            text = f"### Question: {pair['question']}\n### Answer: {pair['answer']}"
            formatted_data.append({"text": text})
        
        # Convert to Dataset
        dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"Created {domain} dataset with {len(dataset)} examples")
        return dataset
    
    def create_classification_dataset(self, task: str = "sentiment") -> Dataset:
        """Create classification dataset"""
        
        if task == "sentiment":
            examples = [
                {"text": "This product is amazing! I love it so much.", "label": 1},
                {"text": "Great quality and fast shipping. Highly recommended.", "label": 1},
                {"text": "Excellent customer service and helpful staff.", "label": 1},
                {"text": "The product broke after one week. Very disappointed.", "label": 0},
                {"text": "Poor quality for the price. Would not buy again.", "label": 0},
                {"text": "Customer service was rude and unhelpful.", "label": 0},
                {"text": "It's okay, nothing special but does the job.", "label": 2},
                {"text": "Average product with decent features.", "label": 2},
                {"text": "Best purchase I've made this year! Outstanding quality.", "label": 1},
                {"text": "Worst experience ever. Complete waste of money.", "label": 0}
            ]
        
        elif task == "intent":
            examples = [
                {"text": "I want to cancel my subscription", "label": 0},  # cancel
                {"text": "How do I update my billing information?", "label": 1},  # account
                {"text": "My order hasn't arrived yet", "label": 2},  # shipping
                {"text": "I need help with the mobile app", "label": 3},  # support
                {"text": "Can you help me find the right product?", "label": 4},  # sales
                {"text": "I'd like to return this item", "label": 0},  # cancel/return
                {"text": "Where can I change my password?", "label": 1},  # account
                {"text": "When will my package be delivered?", "label": 2},  # shipping
                {"text": "The website isn't working properly", "label": 3},  # support
                {"text": "What's the difference between these models?", "label": 4}  # sales
            ]
        
        dataset = Dataset.from_list(examples)
        logger.info(f"Created {task} classification dataset with {len(dataset)} examples")
        return dataset
    
    def create_code_generation_dataset(self) -> Dataset:
        """Create dataset for code generation fine-tuning"""
        
        code_examples = [
            {
                "instruction": "Write a function to calculate the factorial of a number",
                "code": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""
            },
            {
                "instruction": "Create a function to find the maximum element in a list",
                "code": """def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val"""
            },
            {
                "instruction": "Write a function to check if a string is a palindrome",
                "code": """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]"""
            }
        ]
        
        formatted_data = []
        for example in code_examples:
            text = f"### Instruction: {example['instruction']}\n### Code:\n{example['code']}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Created code generation dataset with {len(dataset)} examples")
        return dataset

class LoRAFineTuner:
    """
    Efficient fine-tuning using LoRA (Low-Rank Adaptation)
    
    LoRA Benefits:
    - Dramatically reduces trainable parameters (0.1-1% of original)
    - Faster training and inference
    - Lower memory requirements
    - Easy to swap LoRA adapters for different tasks
    - Preserves base model quality
    """
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.trainer = None
        self.training_metrics = []
        
        # Initialize output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{config.output_dir}/logs", exist_ok=True)
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization if specified"""
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        quantization_config = None
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        
        elif self.config.use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        
        # Load model based on task type
        if self.config.task_type == "SEQ_CLS":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
    
    def setup_lora(self, target_modules: Optional[List[str]] = None):
        """Setup LoRA configuration and wrap model"""
        
        if target_modules is None:
            # Common target modules for different model types
            if "gpt" in self.config.model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in self.config.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "bert" in self.config.model_name.lower():
                target_modules = ["query", "key", "value", "dense"]
            else:
                # Generic fallback - find attention modules
                target_modules = self._find_attention_modules()
        
        # Create LoRA configuration
        task_type = TaskType.CAUSAL_LM if self.config.task_type == "CAUSAL_LM" else TaskType.SEQ_CLS
        
        lora_config = LoraConfig(
            task_type=task_type,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none"  # or "all" if you want to train bias parameters
        )
        
        # Wrap model with LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"LoRA setup complete:")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"  Total parameters: {total_params:,}")
        
        return lora_config
    
    def _find_attention_modules(self) -> List[str]:
        """Automatically find attention modules in the model"""
        
        attention_modules = []
        
        for name, module in self.model.named_modules():
            # Look for common attention module patterns
            if any(pattern in name.lower() for pattern in ['attn', 'attention', 'query', 'key', 'value']):
                # Get the last part of the module name
                module_name = name.split('.')[-1]
                if module_name not in attention_modules:
                    attention_modules.append(module_name)
        
        if not attention_modules:
            # Fallback to linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module_name = name.split('.')[-1]
                    if module_name not in attention_modules:
                        attention_modules.append(module_name)
                        if len(attention_modules) >= 4:  # Limit to avoid too many modules
                            break
        
        logger.info(f"Auto-detected target modules: {attention_modules}")
        return attention_modules
    
    def prepare_dataset(self, dataset: Dataset, task_type: str = "generation") -> Dataset:
        """Prepare dataset for training"""
        
        def tokenize_function_generation(examples):
            """Tokenize for text generation tasks"""
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        def tokenize_function_classification(examples):
            """Tokenize for classification tasks"""
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            tokenized["labels"] = examples["label"]
            return tokenized
        
        # Choose tokenization function based on task
        if task_type == "classification":
            tokenized_dataset = dataset.map(tokenize_function_classification, batched=True)
        else:
            tokenized_dataset = dataset.map(tokenize_function_generation, batched=True)
            # For generation, labels are the same as input_ids
            def add_labels(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None,
              task_type: str = "generation"):
        """Train the model with LoRA"""
        
        if self.peft_model is None:
            self.setup_lora()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset, task_type)
        if eval_dataset:
            eval_dataset = self.prepare_dataset(eval_dataset, task_type)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/checkpoints",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            dataloader_drop_last=True,
            remove_unused_columns=False
        )
        
        # Data collator
        if task_type == "classification":
            data_collator = None  # Default collator for classification
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM
            )
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            if task_type == "classification":
                predictions = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
                return {
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                }
            else:
                # For generation tasks, compute perplexity
                predictions = torch.from_numpy(predictions)
                labels = torch.from_numpy(labels)
                
                # Shift so that tokens < n predict n
                shift_logits = predictions[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                perplexity = torch.exp(loss)
                return {"perplexity": perplexity.item()}
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
        )
        
        # Custom callback to track metrics
        class MetricsCallback:
            def __init__(self, fine_tuner):
                self.fine_tuner = fine_tuner
            
            def on_log(self, logs):
                if logs.get('epoch'):
                    metrics = TrainingMetrics(
                        epoch=logs.get('epoch', 0),
                        train_loss=logs.get('train_loss', 0),
                        eval_loss=logs.get('eval_loss'),
                        perplexity=logs.get('eval_perplexity'),
                        accuracy=logs.get('eval_accuracy'),
                        learning_rate=logs.get('learning_rate', 0)
                    )
                    self.fine_tuner.training_metrics.append(metrics)
        
        # Add callback
        # self.trainer.add_callback(MetricsCallback(self))
        
        # Start training
        logger.info("Starting training...")
        start_time = time.time()
        
        self.trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        self.save_model()
        
        return self.trainer.state.log_history
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the fine-tuned LoRA model"""
        
        if save_path is None:
            save_path = f"{self.config.output_dir}/lora_model"
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config_path = f"{save_path}/training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "model_name": self.config.model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size
            }, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a fine-tuned LoRA model"""
        
        logger.info(f"Loading model from {model_path}")
        
        # Load base model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Model loaded successfully")
    
    def generate_text(self, prompt: str, max_length: int = 200, 
                     temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        """Generate text using the fine-tuned model"""
        
        if self.peft_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt
            text = text[len(prompt):].strip()
            generated_texts.append(text)
        
        return generated_texts
    
    def evaluate_model(self, test_dataset: Dataset, task_type: str = "generation") -> Dict[str, float]:
        """Evaluate the fine-tuned model"""
        
        if self.peft_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        test_dataset = self.prepare_dataset(test_dataset, task_type)
        
        # Use trainer for evaluation
        if self.trainer is None:
            # Create a temporary trainer for evaluation
            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/temp",
                per_device_eval_batch_size=self.config.batch_size,
            )
            
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                tokenizer=self.tokenizer,
            )
        
        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results

class ModelComparison:
    """
    Compare fine-tuned models with base models and RAG systems
    
    Evaluation aspects:
    1. Task-specific performance
    2. Consistency and style
    3. Factual accuracy
    4. Latency and efficiency
    5. Resource requirements
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.comparison_results = []
        
        os.makedirs(f"{output_dir}/comparisons", exist_ok=True)
    
    def compare_models(self, base_model_path: str, finetuned_model_path: str,
                      test_prompts: List[str], task_type: str = "generation") -> Dict[str, Any]:
        """Compare base model vs fine-tuned model performance"""
        
        logger.info("Comparing base model vs fine-tuned model...")
        
        results = {
            "base_model_results": [],
            "finetuned_model_results": [],
            "comparison_metrics": {}
        }
        
        # Load models
        logger.info("Loading base model...")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        logger.info("Loading fine-tuned model...")
        ft_config = FineTuningConfig(model_name=base_model_path)
        fine_tuner = LoRAFineTuner(ft_config)
        fine_tuner.load_model(finetuned_model_path)
        
        # Test prompts
        base_times = []
        ft_times = []
        
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt[:50]}...")
            
            # Base model generation
            start_time = time.time()
            inputs = base_tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                base_output = base_model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=base_tokenizer.eos_token_id
                )
            base_time = time.time() - start_time
            base_text = base_tokenizer.decode(base_output[0], skip_special_tokens=True)
            base_text = base_text[len(prompt):].strip()
            
            base_times.append(base_time)
            results["base_model_results"].append({
                "prompt": prompt,
                "response": base_text,
                "generation_time": base_time
            })
            
            # Fine-tuned model generation
            start_time = time.time()
            ft_texts = fine_tuner.generate_text(prompt, max_length=200)
            ft_time = time.time() - start_time
            
            ft_times.append(ft_time)
            results["finetuned_model_results"].append({
                "prompt": prompt,
                "response": ft_texts[0] if ft_texts else "",
                "generation_time": ft_time
            })
        
        # Calculate comparison metrics
        results["comparison_metrics"] = {
            "base_avg_time": np.mean(base_times),
            "finetuned_avg_time": np.mean(ft_times),
            "time_difference": np.mean(ft_times) - np.mean(base_times),
            "base_model_path": base_model_path,
            "finetuned_model_path": finetuned_model_path,
            "num_test_prompts": len(test_prompts)
        }
        
        # Save results
        comparison_file = f"{self.output_dir}/comparisons/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to {comparison_file}")
        return results
    
    def analyze_fine_tuning_vs_rag(self) -> None:
        """
        Comprehensive analysis of when to use fine-tuning vs RAG
        """
        
        analysis = {
            "Fine-Tuning Advantages": {
                "Style Consistency": "Models learn to consistently apply specific writing styles, tones, or formats",
                "Domain Language": "Better understanding of domain-specific terminology and patterns",
                "Task Optimization": "Models become highly optimized for specific tasks",
                "Latency": "No retrieval step, faster inference",
                "Self-Contained": "No need for external knowledge bases or vector databases"
            },
            
            "Fine-Tuning Disadvantages": {
                "Data Requirements": "Need large amounts of high-quality training data",
                "Knowledge Staleness": "Knowledge is frozen at training time",
                "Computational Cost": "Requires GPU resources and training time",
                "Hallucination Risk": "May generate plausible but incorrect information",
                "Version Management": "Need to retrain for updates"
            },
            
            "RAG Advantages": {
                "Fresh Information": "Always uses up-to-date information from knowledge base",
                "Source Attribution": "Can cite specific sources and evidence",
                "Easy Updates": "Update knowledge by adding/removing documents",
                "Lower Computational": "No training required, just inference",
                "Factual Accuracy": "Grounded in specific retrieved documents"
            },
            
            "RAG Disadvantages": {
                "Retrieval Dependency": "Quality depends on retrieval effectiveness",
                "Context Limitations": "Limited by LLM context window",
                "Latency": "Additional retrieval step adds latency",
                "Style Inconsistency": "May vary in style across different sources",
                "Complex Setup": "Requires vector databases and embedding models"
            },
            
            "Decision Framework": {
                "Choose Fine-Tuning When": [
                    "Need consistent style/tone across all outputs",
                    "Working with structured or domain-specific language patterns",
                    "Have sufficient high-quality training data",
                    "Task is well-defined and stable",
                    "Ultra-low latency is critical",
                    "Knowledge is relatively static"
                ],
                
                "Choose RAG When": [
                    "Knowledge changes frequently",
                    "Need source attribution and transparency",
                    "Working with factual question-answering",
                    "Have limited training data",
                    "Need to combine information from multiple sources",
                    "Want to maintain factual accuracy"
                ]
            },
            
            "Hybrid Approaches": {
                "RAG + Fine-Tuning": "Fine-tune on task format, use RAG for knowledge",
                "Multiple LoRA Adapters": "Different adapters for different domains/tasks",
                "Retrieval-Augmented Fine-Tuning": "Fine-tune using retrieved examples",
                "Adaptive Systems": "Choose approach based on query type"
            }
        }
        
        # Save analysis
        analysis_file = f"{self.output_dir}/comparisons/finetuning_vs_rag_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print key insights
        print("\n🔍 FINE-TUNING vs RAG: DECISION FRAMEWORK")
        print("=" * 60)
        
        print("\n📈 Choose FINE-TUNING when:")
        for reason in analysis["Decision Framework"]["Choose Fine-Tuning When"]:
            print(f"   • {reason}")
        
        print("\n📊 Choose RAG when:")
        for reason in analysis["Decision Framework"]["Choose RAG When"]:
            print(f"   • {reason}")
        
        print(f"\n💡 Hybrid Approaches:")
        for approach, description in analysis["Hybrid Approaches"].items():
            print(f"   • {approach}: {description}")
        
        logger.info(f"Analysis saved to {analysis_file}")

def run_complete_demo():
    """Run comprehensive fine-tuning demonstration"""
    
    print("🚀 LLM FINE-TUNING WITH LORA/PEFT DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows:")
    print("1. Efficient fine-tuning using LoRA (Low-Rank Adaptation)")
    print("2. Comparison with base models and RAG systems")
    print("3. Domain-specific adaptation techniques")
    print("4. Model evaluation and performance analysis")
    print("5. Production deployment considerations")
    print("=" * 80)
    
    # Configuration
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-small",  # Smaller model for demo
        lora_r=16,
        lora_alpha=32,
        learning_rate=3e-4,
        num_epochs=2,  # Reduced for demo
        batch_size=2,  # Smaller batch for limited resources
        output_dir="outputs"
    )
    
    try:
        # 1. Create datasets
        print("\n📚 Creating training datasets...")
        dataset_creator = DatasetCreator()
        
        # Customer support Q&A dataset
        qa_dataset = dataset_creator.create_qa_dataset("customer_support")
        print(f"   Created Q&A dataset: {len(qa_dataset)} examples")
        
        # Split dataset
        train_size = int(0.8 * len(qa_dataset))
        train_dataset = qa_dataset.select(range(train_size))
        eval_dataset = qa_dataset.select(range(train_size, len(qa_dataset)))
        
        # 2. Initialize fine-tuner
        print(f"\n🔧 Initializing LoRA fine-tuner...")
        fine_tuner = LoRAFineTuner(config)
        
        # 3. Setup LoRA
        print(f"\n⚙️ Setting up LoRA configuration...")
        lora_config = fine_tuner.setup_lora()
        
        # 4. Train model
        print(f"\n🏋️ Training model with LoRA...")
        training_history = fine_tuner.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_type="generation"
        )
        
        # 5. Test generation
        print(f"\n💬 Testing model generation...")
        test_prompts = [
            "### Question: How do I reset my password?\n### Answer:",
            "### Question: What is your return policy?\n### Answer:",
            "### Question: How can I track my order?\n### Answer:"
        ]
        
        for prompt in test_prompts:
            question = prompt.split("### Question: ")[1].split("\n### Answer:")[0]
            print(f"\n📝 Q: {question}")
            
            generated = fine_tuner.generate_text(prompt, max_length=150, temperature=0.7)
            print(f"🤖 A: {generated[0] if generated else 'No response generated'}")
        
        # 6. Model comparison
        print(f"\n📊 Analyzing Fine-tuning vs RAG...")
        comparator = ModelComparison(config.output_dir)
        comparator.analyze_fine_tuning_vs_rag()
        
        # 7. Create visualizations
        print(f"\n📈 Creating performance visualizations...")
        create_training_visualizations(training_history, config.output_dir)
        
        # 8. Business impact analysis
        print(f"\n💼 Analyzing business impact...")
        analyze_business_impact(config)
        
        print(f"\n✅ FINE-TUNING DEMO COMPLETE!")
        print(f"📁 Results saved to: {config.output_dir}/")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide fallback analysis
        print(f"\n📚 Providing theoretical analysis instead...")
        comparator = ModelComparison("outputs")
        comparator.analyze_fine_tuning_vs_rag()

def create_training_visualizations(training_history: List[Dict], output_dir: str):
    """Create visualizations of training progress"""
    
    if not training_history:
        logger.warning("No training history available for visualization")
        return
    
    # Extract metrics from training history
    epochs = []
    train_losses = []
    eval_losses = []
    learning_rates = []
    
    for entry in training_history:
        if 'epoch' in entry:
            epochs.append(entry['epoch'])
            train_losses.append(entry.get('train_loss', 0))
            eval_losses.append(entry.get('eval_loss', None))
            learning_rates.append(entry.get('learning_rate', 0))
    
    if not epochs:
        logger.warning("No epoch data found in training history")
        return
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training Loss
    if train_losses:
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if any(loss for loss in eval_losses if loss is not None):
            eval_losses_clean = [loss for loss in eval_losses if loss is not None]
            epochs_eval = epochs[:len(eval_losses_clean)]
            ax1.plot(epochs_eval, eval_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training Progress', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Learning Rate Schedule
    if learning_rates:
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax2.set_title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. LoRA Benefits Illustration
    ax3.bar(['Base Model\nParameters', 'LoRA\nParameters'], 
           [100, 0.5], color=['lightcoral', 'lightgreen'])
    ax3.set_title('Parameter Efficiency with LoRA', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Relative Parameter Count (%)')
    ax3.set_ylim(0, 110)
    
    for i, v in enumerate([100, 0.5]):
        ax3.text(i, v + 2, f'{v}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training Time Comparison (Theoretical)
    methods = ['Full\nFine-tuning', 'LoRA\nFine-tuning', 'RAG\nSetup']
    times = [24, 2, 0.5]  # Hours (theoretical)
    colors = ['red', 'orange', 'green']
    
    bars = ax4.bar(methods, times, color=colors)
    ax4.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Time (Hours)')
    
    for bar, time in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/finetuning_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Training visualizations saved to {output_dir}/finetuning_analysis.png")

def analyze_business_impact(config: FineTuningConfig):
    """Analyze business impact and ROI of fine-tuning"""
    
    print(f"\n💼 BUSINESS IMPACT ANALYSIS: FINE-TUNING")
    print("=" * 60)
    
    # Cost analysis
    training_cost = 50  # Estimated GPU hours * cost per hour
    maintenance_cost = 20  # Monthly model management
    
    # Benefits analysis
    response_quality_improvement = 35  # % improvement in task-specific quality
    consistency_improvement = 60  # % improvement in style consistency
    latency_reduction = 15  # % reduction vs RAG systems
    
    print(f"💰 Cost Analysis:")
    print(f"   Initial Training Cost: ${training_cost}")
    print(f"   Monthly Maintenance: ${maintenance_cost}")
    print(f"   Total 6-month Cost: ${training_cost + (maintenance_cost * 6)}")
    
    print(f"\n📈 Performance Benefits:")
    print(f"   Task-specific Quality: +{response_quality_improvement}%")
    print(f"   Style Consistency: +{consistency_improvement}%")
    print(f"   Response Latency: -{latency_reduction}%")
    
    print(f"\n🎯 Use Cases Where Fine-Tuning Excels:")
    use_cases = [
        "Customer Service: Consistent brand voice and response style",
        "Legal Documents: Domain-specific language and format requirements",
        "Code Generation: Programming language patterns and conventions",
        "Creative Writing: Specific narrative styles and tones",
        "Technical Documentation: Industry terminology and structure"
    ]
    
    for use_case in use_cases:
        print(f"   • {use_case}")
    
    print(f"\n⚖️ Fine-Tuning vs RAG Decision Matrix:")
    print("   ┌─────────────────────┬───────────────┬─────────────┐")
    print("   │ Factor              │ Fine-Tuning   │ RAG         │")
    print("   ├─────────────────────┼───────────────┼─────────────┤")
    print("   │ Setup Cost          │ High          │ Medium      │")
    print("   │ Ongoing Cost        │ Low           │ Medium      │")
    print("   │ Knowledge Updates   │ Expensive     │ Easy        │")
    print("   │ Style Consistency   │ Excellent     │ Variable    │")
    print("   │ Source Attribution  │ None          │ Automatic   │")
    print("   │ Factual Accuracy    │ Risk of drift │ Grounded    │")
    print("   │ Latency             │ Fast          │ Slower      │")
    print("   │ Customization       │ Deep          │ Limited     │")
    print("   └─────────────────────┴───────────────┴─────────────┘")
    
    print(f"\n🚀 Implementation Roadmap:")
    roadmap = [
        "1. Define specific use case and success metrics",
        "2. Collect and curate high-quality training data",
        "3. Start with smaller model for proof of concept",
        "4. Implement LoRA for parameter efficiency",
        "5. Evaluate against base model and alternatives",
        "6. Deploy with monitoring and feedback loops",
        "7. Plan for model updates and maintenance"
    ]
    
    for step in roadmap:
        print(f"   {step}")

if __name__ == "__main__":
    # Run the complete demonstration
    run_complete_demo()
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNING OUTCOMES:")
    print("1. LoRA enables efficient fine-tuning with minimal resources")
    print("2. Fine-tuning excels at style, tone, and domain-specific patterns")
    print("3. RAG is better for factual accuracy and knowledge updates")
    print("4. Hybrid approaches can combine benefits of both methods")
    print("5. Business context determines the optimal approach")
    print("\n🚀 NEXT STEPS:")
    print("• Experiment with different LoRA configurations")
    print("• Try fine-tuning on your own domain-specific data")
    print("• Compare performance with RAG on your use case")
    print("• Implement monitoring and evaluation frameworks")
    print("• Consider hybrid RAG + fine-tuning approaches")
    print("="*80)