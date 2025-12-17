"""
Project 12: NLP Sentiment Analysis Deep Dive
Teaching machines to understand human language

This project explores the fascinating world of Natural Language Processing, from basic
text preprocessing to advanced transformer models. You'll build sentiment analysis
systems that understand context, emotion, and nuance in human language.

Learning Objectives:
1. Master text preprocessing and tokenization techniques
2. Understand different text representation methods (BOW, TF-IDF, embeddings)
3. Implement sentiment analysis from scratch and with pre-trained models
4. Explore the evolution from simple models to transformers
5. Build production-ready text analysis systems

Business Context:
NLP powers modern business intelligence:
- Customer feedback analysis (saving 1000+ hours/month)
- Social media monitoring (real-time brand sentiment)
- Content moderation (automated safety at scale)
- Market research (opinion mining from reviews)
- Customer service automation (intent classification)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
import warnings
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
warnings.filterwarnings('ignore')

# Advanced NLP libraries (install with: pip install transformers torch)
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: Install transformers for advanced models: pip install transformers torch")

# NLTK for advanced text processing (install with: pip install nltk)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Note: Install NLTK for advanced text processing: pip install nltk")

@dataclass
class SentimentResult:
    """Represents a sentiment analysis result"""
    text: str
    predicted_sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float = 0.0

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline
    
    Text preprocessing is crucial for NLP success:
    - Raw text is messy and inconsistent
    - Standardization improves model performance
    - Domain-specific cleaning may be required
    """
    
    def __init__(self, use_nltk: bool = NLTK_AVAILABLE):
        self.use_nltk = use_nltk
        
        if self.use_nltk:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            # Basic stop words if NLTK unavailable
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_clean(self, text: str) -> str:
        """Advanced text cleaning with more sophisticated rules"""
        # Basic cleaning
        text = self.basic_clean(text)
        
        # Handle contractions
        contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would",
            "it'll": "it will", "it's": "it is", "let's": "let us",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what's": "what is",
            "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation but keep emoticons
        text = re.sub(r'[^\w\s:;=\(\)\[\]]+', '', text)
        
        # Handle repeated characters (e.g., "sooooo good" -> "so good")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.use_nltk:
            return word_tokenize(text)
        else:
            # Simple whitespace tokenization
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        if self.use_nltk:
            return [self.stemmer.stem(token) for token in tokens]
        else:
            # Simple stemming (just remove common suffixes)
            suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness']
            stemmed = []
            for token in tokens:
                for suffix in suffixes:
                    if token.endswith(suffix) and len(token) > len(suffix) + 2:
                        token = token[:-len(suffix)]
                        break
                stemmed.append(token)
            return stemmed
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if self.use_nltk:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return tokens  # Fallback to no lemmatization
    
    def preprocess(self, text: str, 
                  remove_stopwords: bool = True,
                  use_stemming: bool = False,
                  use_lemmatization: bool = True) -> str:
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.advanced_clean(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        elif use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis system demonstrating multiple approaches:
    1. Rule-based sentiment (lexicon approach)
    2. Machine learning models (Naive Bayes, Logistic Regression)
    3. Deep learning approaches (if available)
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.preprocessor = TextPreprocessor()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/analysis", exist_ok=True)
        
        # Simple sentiment lexicon (in practice, use VADER or TextBlob)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
            'perfect', 'best', 'awesome', 'brilliant', 'outstanding', 'superb', 'nice',
            'beautiful', 'happy', 'joy', 'pleased', 'satisfied', 'delighted', 'excited',
            'thrilled', 'impressed', 'remarkable', 'exceptional', 'magnificent'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting',
            'annoying', 'disappointing', 'frustrated', 'angry', 'sad', 'upset', 'mad',
            'furious', 'disgusted', 'irritated', 'annoyed', 'pathetic', 'useless',
            'worthless', 'ridiculous', 'stupid', 'crazy', 'insane', 'outrageous'
        }
        
        # Load pre-trained models if available
        self.ml_models = {}
        self.transformers_model = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading pre-trained transformer model...")
                self.transformers_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("✅ Transformer model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Could not load transformer model: {e}")
                self.transformers_model = None
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic sentiment data for demonstration
        
        In real projects, you would use:
        - Customer reviews (Amazon, Yelp, etc.)
        - Social media data (Twitter API)
        - Survey responses
        - Support tickets
        """
        # Positive samples
        positive_templates = [
            "I love this {product}! It's {adjective} and works {adverb}.",
            "This {product} is {adjective}! Highly recommend it.",
            "Amazing {product}! {adjective} quality and {adjective} service.",
            "Best {product} I've ever used. So {adjective}!",
            "Wonderful experience with this {product}. Very {adjective}.",
        ]
        
        # Negative samples
        negative_templates = [
            "This {product} is {adjective}. Total waste of money.",
            "Terrible {product}! {adjective} quality and {adjective} service.",
            "I hate this {product}. It's {adjective} and doesn't work.",
            "Worst {product} ever. Completely {adjective}.",
            "Disappointed with this {product}. Very {adjective}.",
        ]
        
        # Neutral samples
        neutral_templates = [
            "This {product} is okay. Nothing {adjective} about it.",
            "The {product} works as expected. It's {adjective}.",
            "Average {product}. Does the job but nothing {adjective}.",
            "This {product} is fine. Not {adjective}, not {adjective}.",
            "Decent {product}. Could be {adjective} but acceptable.",
        ]
        
        products = ['product', 'service', 'app', 'website', 'tool', 'device', 'software']
        
        positive_adjs = ['amazing', 'excellent', 'fantastic', 'great', 'wonderful', 'perfect']
        negative_adjs = ['terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 'useless']
        neutral_adjs = ['average', 'normal', 'standard', 'typical', 'regular', 'ordinary']
        
        adverbs = ['perfectly', 'smoothly', 'flawlessly', 'efficiently', 'reliably']
        
        data = []
        
        # Generate samples
        for _ in range(n_samples // 3):
            # Positive
            template = np.random.choice(positive_templates)
            text = template.format(
                product=np.random.choice(products),
                adjective=np.random.choice(positive_adjs),
                adverb=np.random.choice(adverbs)
            )
            data.append({'text': text, 'sentiment': 'positive'})
            
            # Negative
            template = np.random.choice(negative_templates)
            text = template.format(
                product=np.random.choice(products),
                adjective=np.random.choice(negative_adjs)
            )
            data.append({'text': text, 'sentiment': 'negative'})
            
            # Neutral
            template = np.random.choice(neutral_templates)
            text = template.format(
                product=np.random.choice(products),
                adjective=np.random.choice(neutral_adjs)
            )
            data.append({'text': text, 'sentiment': 'neutral'})
        
        # Add some real-world variations
        real_examples = [
            {"text": "The customer service was responsive and helpful.", "sentiment": "positive"},
            {"text": "Delivery took longer than expected but product is good.", "sentiment": "neutral"},
            {"text": "Would not recommend this to anyone. Poor quality.", "sentiment": "negative"},
            {"text": "Works as described. No complaints.", "sentiment": "neutral"},
            {"text": "Outstanding performance! Exceeded my expectations!", "sentiment": "positive"},
            {"text": "Complete disaster. Nothing works properly.", "sentiment": "negative"},
        ]
        
        data.extend(real_examples * 10)  # Repeat for more samples
        
        return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    
    def lexicon_based_sentiment(self, text: str) -> SentimentResult:
        """
        Simple lexicon-based sentiment analysis
        
        This approach:
        - Counts positive and negative words
        - Simple but interpretable
        - Works well for basic sentiment
        - Doesn't understand context or sarcasm
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        tokens = processed_text.split()
        
        # Count sentiment words
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Calculate sentiment
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
            pos_score = 0.33
            neg_score = 0.33
            neu_score = 0.34
        else:
            pos_score = positive_count / len(tokens)
            neg_score = negative_count / len(tokens)
            neu_score = 1 - pos_score - neg_score
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = positive_count / total_sentiment_words
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = negative_count / total_sentiment_words
            else:
                sentiment = "neutral"
                confidence = 0.5
        
        return SentimentResult(
            text=text,
            predicted_sentiment=sentiment,
            confidence=confidence,
            positive_score=pos_score,
            negative_score=neg_score,
            neutral_score=neu_score
        )
    
    def train_ml_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train machine learning models for sentiment analysis
        
        We'll train multiple models:
        1. Naive Bayes - works well with text data
        2. Logistic Regression - linear, interpretable
        3. TF-IDF features - captures word importance
        """
        print("🤖 Training Machine Learning Models...")
        
        # Prepare data
        X = df['text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Model 1: Naive Bayes with Count Vectorizer
        print("Training Naive Bayes with Count Vectorizer...")
        nb_pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        # Custom transformer for preprocessing
        class TextPreprocessorTransformer:
            def __init__(self):
                self.preprocessor = TextPreprocessor()
            
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return [self.preprocessor.preprocess(text) for text in X]
        
        # Recreate pipeline with proper transformer
        nb_pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        nb_pipeline.fit(X_train, y_train)
        nb_pred = nb_pipeline.predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_pred)
        
        results['naive_bayes'] = {
            'model': nb_pipeline,
            'accuracy': nb_accuracy,
            'predictions': nb_pred,
            'test_labels': y_test
        }
        
        print(f"   Naive Bayes Accuracy: {nb_accuracy:.3f}")
        
        # Model 2: Logistic Regression with TF-IDF
        print("Training Logistic Regression with TF-IDF...")
        lr_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_pred = lr_pipeline.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        results['logistic_regression'] = {
            'model': lr_pipeline,
            'accuracy': lr_accuracy,
            'predictions': lr_pred,
            'test_labels': y_test
        }
        
        print(f"   Logistic Regression Accuracy: {lr_accuracy:.3f}")
        
        # Save models
        for name, result in results.items():
            model_path = f"{self.output_dir}/models/{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        self.ml_models = results
        return results
    
    def transformer_sentiment(self, text: str) -> SentimentResult:
        """
        Use pre-trained transformer model for sentiment analysis
        
        Transformers represent the state-of-the-art in NLP:
        - Pre-trained on massive datasets
        - Understand context and nuance
        - Handle sarcasm and complex language better
        - More computationally expensive
        """
        if not self.transformers_model:
            return SentimentResult(
                text=text,
                predicted_sentiment="neutral",
                confidence=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0
            )
        
        try:
            # Get predictions
            results = self.transformers_model(text)[0]
            
            # Parse results (format may vary by model)
            scores = {result['label'].lower(): result['score'] for result in results}
            
            # Map labels (different models use different labels)
            label_mapping = {
                'label_0': 'negative',
                'label_1': 'neutral', 
                'label_2': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'positive': 'positive'
            }
            
            mapped_scores = {}
            for label, score in scores.items():
                mapped_label = label_mapping.get(label, label)
                mapped_scores[mapped_label] = score
            
            # Determine prediction
            predicted_sentiment = max(mapped_scores, key=mapped_scores.get)
            confidence = mapped_scores[predicted_sentiment]
            
            return SentimentResult(
                text=text,
                predicted_sentiment=predicted_sentiment,
                confidence=confidence,
                positive_score=mapped_scores.get('positive', 0.0),
                negative_score=mapped_scores.get('negative', 0.0),
                neutral_score=mapped_scores.get('neutral', 0.0)
            )
        
        except Exception as e:
            print(f"Transformer prediction failed: {e}")
            return SentimentResult(
                text=text,
                predicted_sentiment="neutral",
                confidence=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0
            )
    
    def analyze_text_batch(self, texts: List[str], method: str = "all") -> pd.DataFrame:
        """Analyze a batch of texts with different methods"""
        results = []
        
        for text in texts:
            result = {'text': text[:50] + '...' if len(text) > 50 else text}
            
            if method in ["all", "lexicon"]:
                lexicon_result = self.lexicon_based_sentiment(text)
                result['lexicon_sentiment'] = lexicon_result.predicted_sentiment
                result['lexicon_confidence'] = lexicon_result.confidence
            
            if method in ["all", "ml"] and 'logistic_regression' in self.ml_models:
                ml_pred = self.ml_models['logistic_regression']['model'].predict([text])[0]
                ml_proba = self.ml_models['logistic_regression']['model'].predict_proba([text])[0]
                result['ml_sentiment'] = ml_pred
                result['ml_confidence'] = max(ml_proba)
            
            if method in ["all", "transformer"]:
                transformer_result = self.transformer_sentiment(text)
                result['transformer_sentiment'] = transformer_result.predicted_sentiment
                result['transformer_confidence'] = transformer_result.confidence
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def visualize_results(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Create comprehensive visualizations of sentiment analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution in data
        sentiment_counts = df['sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                      colors=['#ff6b6b', '#feca57', '#48cae4'])
        axes[0, 0].set_title('Sentiment Distribution in Dataset', fontweight='bold')
        
        # 2. Model accuracy comparison
        if self.ml_models:
            model_names = list(self.ml_models.keys())
            accuracies = [self.ml_models[name]['accuracy'] for name in model_names]
            
            bars = axes[0, 1].bar(model_names, accuracies, color=['#6c5ce7', '#74b9ff'])
            axes[0, 1].set_title('Model Accuracy Comparison', fontweight='bold')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confusion matrix for best model
        if self.ml_models:
            best_model = max(self.ml_models.keys(), key=lambda k: self.ml_models[k]['accuracy'])
            cm = confusion_matrix(
                self.ml_models[best_model]['test_labels'],
                self.ml_models[best_model]['predictions']
            )
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
            axes[0, 2].set_title(f'Confusion Matrix - {best_model.title()}', fontweight='bold')
            axes[0, 2].set_xlabel('Predicted')
            axes[0, 2].set_ylabel('Actual')
        
        # 4. Word frequency analysis
        all_text = ' '.join(df['text'])
        processed_text = self.preprocessor.preprocess(all_text)
        words = processed_text.split()
        word_freq = Counter(words).most_common(15)
        
        words_list, counts = zip(*word_freq)
        axes[1, 0].barh(range(len(words_list)), counts, color='#a29bfe')
        axes[1, 0].set_yticks(range(len(words_list)))
        axes[1, 0].set_yticklabels(words_list)
        axes[1, 0].set_title('Top 15 Most Frequent Words', fontweight='bold')
        axes[1, 0].set_xlabel('Frequency')
        
        # 5. Sentiment by text length
        df['text_length'] = df['text'].str.len()
        sentiment_by_length = df.groupby('sentiment')['text_length'].mean()
        
        bars = axes[1, 1].bar(sentiment_by_length.index, sentiment_by_length.values,
                             color=['#ff6b6b', '#feca57', '#48cae4'])
        axes[1, 1].set_title('Average Text Length by Sentiment', fontweight='bold')
        axes[1, 1].set_ylabel('Average Text Length (characters)')
        
        # Add value labels
        for bar, length in zip(bars, sentiment_by_length.values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{length:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Model comparison on sample texts
        sample_texts = [
            "This product is absolutely amazing!",
            "Terrible quality, waste of money.",
            "It's okay, nothing special.",
            "Best purchase ever, highly recommend!",
            "Disappointed with the service."
        ]
        
        comparison_results = self.analyze_text_batch(sample_texts, method="all")
        
        # Create a heatmap of confidence scores
        if 'lexicon_confidence' in comparison_results.columns:
            confidence_data = comparison_results[
                ['lexicon_confidence', 'ml_confidence', 'transformer_confidence']
            ].fillna(0)
            
            sns.heatmap(confidence_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                       xticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
                       yticklabels=['Lexicon', 'ML Model', 'Transformer'],
                       ax=axes[1, 2])
            axes[1, 2].set_title('Method Confidence Comparison', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/comprehensive_sentiment_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def business_impact_analysis(self, df: pd.DataFrame):
        """
        Analyze business impact of sentiment analysis
        
        This demonstrates how sentiment analysis creates business value:
        - Customer satisfaction tracking
        - Brand reputation monitoring
        - Product feedback analysis
        - Resource allocation insights
        """
        print("\n💼 BUSINESS IMPACT ANALYSIS")
        print("=" * 60)
        
        # Calculate key business metrics
        total_reviews = len(df)
        positive_rate = (df['sentiment'] == 'positive').mean()
        negative_rate = (df['sentiment'] == 'negative').mean()
        neutral_rate = (df['sentiment'] == 'neutral').mean()
        
        print(f"📊 Customer Sentiment Overview:")
        print(f"   Total Reviews Analyzed: {total_reviews:,}")
        print(f"   Positive Sentiment: {positive_rate:.1%}")
        print(f"   Negative Sentiment: {negative_rate:.1%}")
        print(f"   Neutral Sentiment: {neutral_rate:.1%}")
        
        # Business scenarios
        scenarios = [
            {
                "name": "E-commerce Platform",
                "volume": 10000,
                "value_per_review": 50,
                "manual_cost_per_review": 2.5
            },
            {
                "name": "SaaS Product",
                "volume": 2500,
                "value_per_review": 200,
                "manual_cost_per_review": 5.0
            },
            {
                "name": "Restaurant Chain", 
                "volume": 50000,
                "value_per_review": 25,
                "manual_cost_per_review": 1.0
            }
        ]
        
        print(f"\n💰 ROI Analysis for Different Business Scenarios:")
        
        for scenario in scenarios:
            print(f"\n📈 {scenario['name']}:")
            print(f"   Monthly Review Volume: {scenario['volume']:,}")
            
            # Cost savings calculation
            manual_cost = scenario['volume'] * scenario['manual_cost_per_review']
            automation_cost = 5000  # Monthly automation cost
            cost_savings = manual_cost - automation_cost
            
            print(f"   Manual Analysis Cost: ${manual_cost:,.0f}/month")
            print(f"   Automation Cost: ${automation_cost:,.0f}/month")
            print(f"   Monthly Savings: ${cost_savings:,.0f}")
            print(f"   Annual Savings: ${cost_savings * 12:,.0f}")
            
            # Revenue impact calculation
            positive_reviews = scenario['volume'] * positive_rate
            negative_reviews = scenario['volume'] * negative_rate
            
            # Assume each positive review increases customer lifetime value
            # and each negative review caught early prevents churn
            positive_value = positive_reviews * scenario['value_per_review'] * 0.1  # 10% boost
            negative_value_saved = negative_reviews * scenario['value_per_review'] * 0.5  # Prevent 50% churn
            
            total_value = positive_value + negative_value_saved
            
            print(f"   Value from Positive Insights: ${positive_value:,.0f}/month")
            print(f"   Value from Issue Prevention: ${negative_value_saved:,.0f}/month")
            print(f"   Total Business Value: ${total_value:,.0f}/month")
            print(f"   ROI: {((cost_savings + total_value) / automation_cost - 1) * 100:.0f}%")
        
        # Actionable insights
        print(f"\n🎯 ACTIONABLE BUSINESS INSIGHTS:")
        
        if negative_rate > 0.3:
            print("⚠️  HIGH NEGATIVE SENTIMENT DETECTED:")
            print("   • Immediate action required on customer experience")
            print("   • Review product quality and service processes")
            print("   • Implement rapid response team for negative feedback")
        
        if positive_rate > 0.6:
            print("✅ STRONG POSITIVE SENTIMENT:")
            print("   • Leverage positive reviews for marketing")
            print("   • Identify and replicate success factors")
            print("   • Consider premium pricing strategies")
        
        if neutral_rate > 0.5:
            print("📈 OPPORTUNITY FOR DIFFERENTIATION:")
            print("   • Many customers are neutral - room for improvement")
            print("   • Focus on creating 'wow' moments")
            print("   • Invest in unique value propositions")
        
        # Implementation roadmap
        print(f"\n🚀 IMPLEMENTATION ROADMAP:")
        print("Phase 1 (Month 1-2): Basic sentiment monitoring setup")
        print("   • Deploy automated sentiment analysis pipeline")
        print("   • Set up real-time alerts for negative sentiment spikes")
        print("   • Train customer service team on insights interpretation")
        
        print("\nPhase 2 (Month 3-4): Advanced analytics and integration")
        print("   • Integrate with CRM and support systems")
        print("   • Develop customer satisfaction prediction models")
        print("   • Create executive dashboards and KPI tracking")
        
        print("\nPhase 3 (Month 5-6): Proactive optimization")
        print("   • Implement predictive churn prevention")
        print("   • Personalize customer experience based on sentiment")
        print("   • Optimize product development using sentiment insights")
    
    def run_complete_analysis(self):
        """Run the complete NLP sentiment analysis demonstration"""
        print("🧠 NLP SENTIMENT ANALYSIS: TEACHING MACHINES TO UNDERSTAND LANGUAGE")
        print("=" * 80)
        print("This comprehensive analysis demonstrates:")
        print("1. Text preprocessing and tokenization techniques")
        print("2. Multiple approaches: lexicon-based, ML, and transformers")
        print("3. Model comparison and evaluation")
        print("4. Business impact analysis and ROI calculations")
        print("5. Real-world implementation strategies")
        print("=" * 80)
        
        try:
            # 1. Generate sample data
            print("\n1. 📝 Generating sample sentiment data...")
            df = self.generate_sample_data(1000)
            print(f"   Generated {len(df)} samples")
            print(f"   Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
            
            # Save sample data
            df.to_csv(f"{self.output_dir}/analysis/sample_sentiment_data.csv", index=False)
            
            # 2. Train ML models
            print("\n2. 🤖 Training machine learning models...")
            ml_results = self.train_ml_models(df)
            
            # 3. Demonstrate different approaches
            print("\n3. 🔍 Demonstrating different sentiment analysis approaches...")
            
            test_examples = [
                "This product is absolutely fantastic! I love everything about it.",
                "Terrible experience. The worst service I've ever encountered.",
                "It's okay, nothing special but does the job.",
                "Amazing quality and fast delivery. Highly recommend!",
                "Not happy with this purchase. Very disappointed.",
                "The app works fine but could use some improvements.",
                "Outstanding customer service! They went above and beyond.",
                "Complete waste of money. Doesn't work as advertised."
            ]
            
            print("\n   Analyzing sample texts with all methods...")
            comparison_df = self.analyze_text_batch(test_examples, method="all")
            print("\n   Sample Results:")
            print(comparison_df.to_string())
            
            # Save detailed results
            comparison_df.to_csv(f"{self.output_dir}/analysis/method_comparison.csv", index=False)
            
            # 4. Create visualizations
            print("\n4. 📊 Creating comprehensive visualizations...")
            self.visualize_results(df, ml_results)
            
            # 5. Business impact analysis
            self.business_impact_analysis(df)
            
            # 6. Model interpretability analysis
            self.analyze_model_interpretability(ml_results)
            
            print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
            print(f"📁 All results saved to: {self.output_dir}/")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_model_interpretability(self, ml_results: Dict[str, Any]):
        """Analyze what the models learned - feature importance and word weights"""
        print("\n🔍 MODEL INTERPRETABILITY ANALYSIS")
        print("=" * 50)
        
        if 'logistic_regression' not in ml_results:
            print("No trained models available for interpretability analysis.")
            return
        
        # Get the trained pipeline
        lr_pipeline = ml_results['logistic_regression']['model']
        vectorizer = lr_pipeline.named_steps['vectorizer']
        classifier = lr_pipeline.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for each class
        classes = classifier.classes_
        coefficients = classifier.coef_
        
        print(f"Model learned from {len(feature_names)} features (words/ngrams)")
        print(f"Classes: {list(classes)}")
        
        # For each class, find most important words
        for i, class_name in enumerate(classes):
            print(f"\n📝 Most Important Words for '{class_name.upper()}' sentiment:")
            
            if len(classes) > 2:  # Multi-class
                class_coef = coefficients[i]
            else:  # Binary classification
                class_coef = coefficients[0] if class_name == classes[1] else -coefficients[0]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(class_coef)[-10:][::-1]
            top_negative_idx = np.argsort(class_coef)[:10]
            
            print("   Positive indicators:")
            for idx in top_positive_idx:
                print(f"     • {feature_names[idx]}: {class_coef[idx]:.3f}")
            
            print("   Negative indicators:")
            for idx in top_negative_idx:
                print(f"     • {feature_names[idx]}: {class_coef[idx]:.3f}")

if __name__ == "__main__":
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNING OUTCOMES:")
    print("1. Text preprocessing is crucial for NLP model performance")
    print("2. Different approaches (lexicon, ML, transformers) have different strengths")
    print("3. Business value comes from actionable insights, not just accuracy")
    print("4. Model interpretability helps build trust and improve systems")
    print("5. Implementation requires considering costs, scalability, and ROI")
    print("\n🚀 NEXT STEPS:")
    print("• Experiment with different preprocessing techniques")
    print("• Try domain-specific sentiment lexicons")
    print("• Implement real-time sentiment monitoring")
    print("• Explore multi-lingual sentiment analysis")
    print("• Build custom models for specific business domains")
    print("="*80)