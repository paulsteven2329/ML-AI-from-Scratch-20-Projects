"""
Project 15: RAG System - Chat with Your Own PDFs
"Why RAG beats fine-tuning in most real cases"

This project implements a complete Retrieval-Augmented Generation (RAG) system
that allows you to chat with your own documents. You'll learn why RAG is often
superior to fine-tuning for knowledge-intensive tasks.

Learning Objectives:
1. Understand RAG architecture and its advantages
2. Implement document chunking and embedding strategies
3. Build vector databases for semantic search
4. Create context-aware LLM applications
5. Evaluate and optimize RAG system performance

Business Context:
RAG systems power modern knowledge management:
- Enterprise search: Find answers in company documents instantly
- Customer support: Automated responses from product manuals
- Legal research: Query case law and regulations efficiently
- Medical diagnosis: Reference latest research and guidelines
- Education: Interactive learning from textbook content
"""

import os
import json
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF libraries not available. Install with: pip install PyPDF2 pdfplumber")

# Vector embeddings and search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    logger.warning("Vector libraries not available. Install with: pip install sentence-transformers faiss-cpu")

# LLM integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Install with: pip install transformers torch")

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    content: str
    source: str
    page_number: int
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

@dataclass 
class RAGResponse:
    """Structured response from RAG system"""
    question: str
    answer: str
    source_chunks: List[DocumentChunk]
    confidence_score: float
    processing_time: float
    retrieval_scores: List[float]

class DocumentProcessor:
    """
    Advanced document processing for RAG systems
    
    Key considerations:
    - Chunk size: Balance between context and relevance
    - Overlap: Ensure important information isn't lost at boundaries
    - Metadata: Preserve document structure and context
    - Quality: Clean and preprocess text for better embeddings
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_sample_documents(self) -> List[Dict[str, str]]:
        """
        Create sample documents for RAG demonstration
        
        In production, these would be loaded from:
        - PDF files using PyPDF2/pdfplumber
        - Word documents using python-docx
        - Web scraping using BeautifulSoup
        - Database exports
        - API integrations
        """
        
        sample_docs = [
            {
                "title": "Machine Learning Best Practices Guide",
                "content": """
Machine Learning Best Practices Guide

Chapter 1: Data Preparation
Data preparation is often considered the most crucial step in any machine learning project, typically consuming 70-80% of a data scientist's time. Proper data preparation involves several key steps:

1. Data Collection and Integration
- Identify all relevant data sources
- Establish data pipelines for continuous integration
- Ensure data quality and consistency across sources
- Handle different data formats and schemas

2. Data Cleaning and Preprocessing
- Handle missing values through imputation or removal
- Detect and treat outliers using statistical methods
- Normalize and standardize numerical features
- Encode categorical variables appropriately
- Remove duplicates and inconsistencies

3. Feature Engineering
Feature engineering is the process of creating new features from existing data to improve model performance. Effective feature engineering techniques include:
- Creating polynomial features for non-linear relationships
- Binning continuous variables into categories
- Creating interaction features between variables
- Time-based features for temporal data
- Domain-specific transformations

Chapter 2: Model Selection and Training
Choosing the right algorithm depends on several factors including data size, problem type, and interpretability requirements.

For classification problems:
- Logistic Regression: Good baseline, interpretable
- Random Forest: Handles non-linear relationships well
- Support Vector Machines: Effective for high-dimensional data
- Neural Networks: Powerful for complex patterns
- Gradient Boosting: Often provides best performance

For regression problems:
- Linear Regression: Simple and interpretable baseline
- Random Forest Regressor: Robust to outliers
- XGBoost: Often wins competitions
- Neural Networks: For complex non-linear relationships

Chapter 3: Model Evaluation and Validation
Proper evaluation prevents overfitting and ensures generalization:

Cross-Validation Strategies:
- K-fold cross-validation for general use
- Stratified cross-validation for imbalanced data
- Time series split for temporal data
- Leave-one-out for small datasets

Evaluation Metrics:
- Classification: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Regression: MAE, MSE, RMSE, R-squared
- Business metrics: Revenue impact, customer satisfaction

Chapter 4: Deployment and Monitoring
Model deployment requires careful consideration of:
- Scalability: Can the model handle production load?
- Latency: Does it meet response time requirements?
- Monitoring: How to detect model drift and performance degradation?
- Maintenance: How to retrain and update models?

Best practices for deployment:
- A/B testing for gradual rollout
- Canary deployments to minimize risk
- Real-time monitoring of key metrics
- Automated retraining pipelines
- Fallback strategies for model failures
"""
            },
            {
                "title": "Deep Learning Architecture Guide",
                "content": """
Deep Learning Architecture Guide

Introduction to Neural Network Architectures
Deep learning has revolutionized artificial intelligence through the development of sophisticated neural network architectures. Understanding when and how to use different architectures is crucial for success.

Feedforward Neural Networks
Feedforward networks are the foundation of deep learning:
- Input layer receives data
- Hidden layers perform transformations
- Output layer produces predictions
- Information flows in one direction

Key considerations for feedforward networks:
- Number of hidden layers (depth)
- Number of neurons per layer (width)
- Activation functions (ReLU, sigmoid, tanh)
- Regularization techniques (dropout, L1/L2)

Convolutional Neural Networks (CNNs)
CNNs excel at processing grid-like data such as images:

Core Components:
- Convolutional layers: Apply filters to detect features
- Pooling layers: Reduce spatial dimensions
- Fully connected layers: Final classification/regression

Popular CNN Architectures:
- LeNet: Early CNN for digit recognition
- AlexNet: Breakthrough in image classification
- VGG: Demonstrated power of depth
- ResNet: Solved vanishing gradient problem with skip connections
- EfficientNet: Optimized for efficiency

Applications:
- Image classification and object detection
- Medical imaging analysis
- Computer vision in autonomous vehicles
- Style transfer and image generation

Recurrent Neural Networks (RNNs)
RNNs process sequential data by maintaining internal state:

Types of RNNs:
- Vanilla RNN: Simple but suffers from vanishing gradients
- LSTM: Uses gates to control information flow
- GRU: Simplified version of LSTM
- Bidirectional RNNs: Process sequences in both directions

Applications:
- Natural language processing
- Time series forecasting
- Speech recognition
- Machine translation

Transformer Architecture
Transformers have revolutionized NLP and are expanding to other domains:

Key Innovations:
- Self-attention mechanism
- Parallel processing of sequences
- Positional encoding for sequence order
- Multi-head attention for different types of relationships

Transformer Variants:
- BERT: Bidirectional encoder for understanding
- GPT: Autoregressive decoder for generation
- T5: Text-to-text unified framework
- Vision Transformer: Adapting transformers to images

Attention Mechanism
Attention allows models to focus on relevant parts of input:
- Scaled dot-product attention
- Multi-head attention for parallel processing
- Self-attention for relating different positions
- Cross-attention for encoder-decoder architectures

Generative Models
Generative models learn to create new data:

Variational Autoencoders (VAEs):
- Learn compressed representations
- Generate new samples from latent space
- Useful for data augmentation

Generative Adversarial Networks (GANs):
- Two networks competing against each other
- Generator creates fake data
- Discriminator distinguishes real from fake
- Applications in image synthesis and style transfer

Architecture Selection Guidelines
Choosing the right architecture depends on:

Problem Type:
- Computer vision: Start with CNNs
- Sequential data: Consider RNNs or Transformers
- Tabular data: Feedforward networks often sufficient
- Generation tasks: GANs or VAEs

Data Characteristics:
- Large datasets: Deeper networks often better
- Small datasets: Risk of overfitting, use regularization
- High-dimensional data: Consider dimensionality reduction

Computational Resources:
- Limited resources: MobileNet, EfficientNet
- Abundant resources: Large transformers, ensembles
- Real-time inference: Optimize for speed

Best Practices for Deep Learning
1. Start simple and gradually increase complexity
2. Use pre-trained models when possible (transfer learning)
3. Implement proper data augmentation
4. Monitor training with validation metrics
5. Use early stopping to prevent overfitting
6. Experiment with different optimizers and learning rates
7. Visualize learned features to understand model behavior
8. Consider ensemble methods for improved performance
"""
            },
            {
                "title": "AI Ethics and Responsible Development",
                "content": """
AI Ethics and Responsible Development

The Importance of Ethical AI
As artificial intelligence becomes increasingly integrated into our daily lives, the need for ethical considerations has never been more critical. AI systems can significantly impact society, affecting everything from employment opportunities to criminal justice decisions.

Key Ethical Principles:
- Fairness and non-discrimination
- Transparency and explainability
- Privacy and data protection
- Accountability and responsibility
- Human autonomy and dignity

Bias in AI Systems
AI bias can manifest in various forms and stages:

Sources of Bias:
- Historical bias in training data
- Representation bias (underrepresented groups)
- Measurement bias (how data is collected)
- Evaluation bias (inappropriate metrics)
- Deployment bias (different contexts)

Types of Bias:
- Algorithmic bias: Built into the algorithm itself
- Data bias: Stemming from biased training data
- Confirmation bias: Reinforcing existing prejudices
- Selection bias: Non-representative samples

Mitigation Strategies:
- Diverse and representative datasets
- Bias detection and measurement tools
- Algorithmic debiasing techniques
- Diverse development teams
- Regular auditing and monitoring

Privacy and Data Protection
AI systems often require large amounts of personal data:

Privacy Concerns:
- Data collection without explicit consent
- Re-identification of anonymized data
- Inference of sensitive attributes
- Data sharing with third parties
- Long-term storage and usage

Privacy-Preserving Techniques:
- Differential privacy: Adding mathematical noise
- Federated learning: Training without centralizing data
- Homomorphic encryption: Computing on encrypted data
- Secure multi-party computation
- Data minimization principles

Transparency and Explainability
The "black box" nature of many AI systems raises concerns:

Why Explainability Matters:
- Building trust with users
- Regulatory compliance requirements
- Debugging and improving models
- Detecting bias and discrimination
- Human oversight and control

Approaches to Explainability:
- Local explanations: Why this specific prediction?
- Global explanations: How does the model work overall?
- Example-based explanations: Similar cases
- Counterfactual explanations: What would change the outcome?

Explainable AI Tools:
- LIME: Local interpretable model-agnostic explanations
- SHAP: Unified framework for feature importance
- Integrated Gradients: Attribution method for deep networks
- Attention visualization for transformers

Accountability and Governance
Establishing responsibility for AI decisions:

Governance Frameworks:
- Clear roles and responsibilities
- Risk assessment and management
- Audit trails and documentation
- Human oversight mechanisms
- Incident response procedures

Regulatory Landscape:
- GDPR: Right to explanation
- AI Act (EU): Risk-based regulation
- Algorithmic accountability acts
- Industry-specific regulations
- Professional ethics codes

Social Impact and Fairness
AI systems can perpetuate or amplify societal inequalities:

Areas of Concern:
- Hiring and recruitment algorithms
- Criminal justice risk assessment
- Healthcare diagnosis and treatment
- Financial services and credit scoring
- Educational assessment and placement

Fairness Metrics:
- Demographic parity: Equal outcomes across groups
- Equalized odds: Equal error rates across groups
- Calibration: Predictions equally accurate across groups
- Individual fairness: Similar individuals treated similarly

Human-AI Collaboration
Designing AI systems that augment rather than replace humans:

Principles of Human-AI Collaboration:
- Human-in-the-loop systems
- Meaningful human control
- AI as decision support tool
- Preserving human agency
- Continuous learning and adaptation

Best Practices:
- Clear indication when AI is involved
- Easy override mechanisms
- Continuous monitoring and feedback
- Training for human operators
- Regular system updates and improvements

Responsible Development Practices
Building ethics into the development process:

Development Lifecycle:
1. Problem definition: Consider societal impact
2. Data collection: Ensure representative and fair datasets
3. Model development: Include fairness constraints
4. Testing: Comprehensive bias and robustness testing
5. Deployment: Gradual rollout with monitoring
6. Monitoring: Continuous performance and fairness tracking

Organizational Practices:
- Diverse and interdisciplinary teams
- Ethics review boards
- Regular ethics training
- External audits and assessments
- Stakeholder engagement and feedback

Tools and Frameworks:
- AI ethics checklists
- Bias detection toolkits
- Fairness-aware machine learning libraries
- Risk assessment frameworks
- Impact evaluation methodologies

Future Considerations
Emerging ethical challenges in AI:

Advanced AI Systems:
- Artificial General Intelligence (AGI) safety
- Autonomous weapons systems
- AI-generated content and deepfakes
- Quantum computing and AI
- Brain-computer interfaces

Global Cooperation:
- International AI governance standards
- Cross-border data sharing agreements
- Harmonized ethical frameworks
- Technology transfer considerations
- Digital divide and AI accessibility

Conclusion
Ethical AI development is not just a moral imperative but also a business necessity. Organizations that prioritize ethical considerations will build more robust, trustworthy, and sustainable AI systems that benefit society as a whole.

Key Takeaways:
- Ethics should be considered throughout the AI lifecycle
- Bias and fairness require ongoing attention and mitigation
- Transparency and explainability build trust and enable oversight
- Human-AI collaboration should preserve human agency
- Regulatory compliance is increasingly important
- Responsible development practices are essential for long-term success
"""
            }
        ]
        
        return sample_docs
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF file with metadata"""
        
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available")
        
        text = ""
        metadata = {"pages": 0, "title": "", "author": "", "creation_date": ""}
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract metadata if available
                if pdf.metadata:
                    metadata.update({
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "creation_date": str(pdf.metadata.get("CreationDate", ""))
                    })
        
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Extract metadata
                    if pdf_reader.metadata:
                        metadata.update({
                            "title": pdf_reader.metadata.get("/Title", ""),
                            "author": pdf_reader.metadata.get("/Author", ""),
                            "creation_date": str(pdf_reader.metadata.get("/CreationDate", ""))
                        })
            
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                raise
        
        return text, metadata
    
    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Intelligent text chunking for optimal retrieval
        
        Strategies implemented:
        1. Sentence-aware chunking: Don't break sentences
        2. Paragraph preservation: Keep related content together
        3. Overlap handling: Ensure context continuity
        4. Metadata preservation: Track source and structure
        """
        
        chunks = []
        
        # Clean and preprocess text
        text = self._clean_text(text)
        
        # Split into sentences for smarter chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_chunk_index = 0
        current_page = 1
        
        for sentence in sentences:
            # Check for page markers
            page_match = re.search(r'--- Page (\d+) ---', sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    source=source,
                    page_number=current_page,
                    chunk_index=current_chunk_index,
                    metadata={"word_count": len(current_chunk.split())}
                )
                chunks.append(chunk)
                
                # Handle overlap
                if self.overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                current_chunk_index += 1
            else:
                current_chunk += " " + sentence
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source=source,
                page_number=current_page,
                chunk_index=current_chunk_index,
                metadata={"word_count": len(current_chunk.split())}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'\n.*?Page \d+.*?\n', '\n', text)
        text = re.sub(r'\n.*?\d{1,2}/\d{1,2}/\d{2,4}.*?\n', '\n', text)
        
        # Fix common OCR errors
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after sentence endings
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        
        return text.strip()
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Extract overlap text from the end of a chunk"""
        
        words = text.split()
        if len(words) <= overlap_size:
            return text
        
        return " ".join(words[-overlap_size:])

class VectorDatabase:
    """
    Vector database for semantic search and retrieval
    
    Implements efficient similarity search using:
    - Dense embeddings for semantic understanding
    - FAISS for fast approximate nearest neighbor search
    - Metadata filtering for precise results
    - Caching for improved performance
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 dimension: int = 384):
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Initialize embedding model
        if VECTOR_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        else:
            self.embedding_model = None
            logger.warning("Vector libraries not available - using mock embeddings")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector database"""
        
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        
        if self.embedding_model:
            # Real embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = embeddings.astype('float32')
        else:
            # Mock embeddings for testing
            embeddings = np.random.rand(len(texts), self.dimension).astype('float32')
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to collection
        self.chunks.extend(chunks)
        
        # Update FAISS index
        self._update_index()
        
        logger.info(f"Vector database now contains {len(self.chunks)} chunks")
    
    def _update_index(self) -> None:
        """Update FAISS index with all embeddings"""
        
        if not self.chunks:
            return
        
        # Stack all embeddings
        embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
        
        if VECTOR_AVAILABLE:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
        else:
            # Store embeddings for mock search
            self.embeddings = embeddings
        
        logger.info(f"Updated FAISS index with {len(embeddings)} embeddings")
    
    def search(self, query: str, top_k: int = 5, 
               filter_source: Optional[str] = None) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Search for relevant chunks using semantic similarity
        
        Returns:
            - List of relevant chunks
            - List of similarity scores
        """
        
        if not self.chunks:
            return [], []
        
        # Generate query embedding
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
        else:
            # Mock embedding
            query_embedding = np.random.rand(1, self.dimension).astype('float32')
        
        # Search using FAISS or manual similarity
        if VECTOR_AVAILABLE and self.index:
            # FAISS search
            scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))
            
            # Convert to lists
            scores = scores[0].tolist()
            indices = indices[0].tolist()
            
            # Get corresponding chunks
            results = []
            result_scores = []
            
            for score, idx in zip(scores, indices):
                if idx >= 0 and idx < len(self.chunks):  # Valid index
                    chunk = self.chunks[idx]
                    
                    # Apply source filter if specified
                    if filter_source is None or chunk.source == filter_source:
                        results.append(chunk)
                        result_scores.append(score)
                        
                        if len(results) >= top_k:
                            break
        
        else:
            # Manual similarity calculation (fallback)
            similarities = []
            
            for chunk in self.chunks:
                # Simple cosine similarity
                sim = np.dot(query_embedding[0], chunk.embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk.embedding)
                )
                similarities.append(sim)
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            result_scores = []
            
            for idx in top_indices:
                chunk = self.chunks[idx]
                score = similarities[idx]
                
                # Apply source filter if specified
                if filter_source is None or chunk.source == filter_source:
                    results.append(chunk)
                    result_scores.append(score)
        
        return results, result_scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        if not self.chunks:
            return {"total_chunks": 0}
        
        sources = [chunk.source for chunk in self.chunks]
        pages = [chunk.page_number for chunk in self.chunks]
        word_counts = [chunk.metadata.get("word_count", 0) for chunk in self.chunks if chunk.metadata]
        
        from collections import Counter
        
        return {
            "total_chunks": len(self.chunks),
            "unique_sources": len(set(sources)),
            "source_distribution": dict(Counter(sources)),
            "page_range": (min(pages) if pages else 0, max(pages) if pages else 0),
            "avg_chunk_words": np.mean(word_counts) if word_counts else 0,
            "total_words": sum(word_counts),
            "embedding_dimension": self.dimension
        }

class RAGSystem:
    """
    Complete Retrieval-Augmented Generation system
    
    Combines:
    1. Document processing and chunking
    2. Vector database for retrieval  
    3. LLM for generation
    4. Context management and optimization
    5. Performance monitoring and evaluation
    """
    
    def __init__(self, llm_provider: str = "openai", embedding_model: str = "all-MiniLM-L6-v2",
                 output_dir: str = "outputs"):
        
        self.processor = DocumentProcessor()
        self.vector_db = VectorDatabase(embedding_model)
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.conversation_history = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/conversations", exist_ok=True)
        os.makedirs(f"{output_dir}/evaluations", exist_ok=True)
        
        # Initialize LLM
        self._init_llm()
    
    def _init_llm(self) -> None:
        """Initialize LLM for generation"""
        
        if self.llm_provider.lower() == "openai":
            if OPENAI_AVAILABLE and ("OPENAI_API_KEY" in os.environ or hasattr(openai, 'api_key')):
                self.use_real_llm = True
                logger.info("OpenAI LLM initialized")
            else:
                self.use_real_llm = False
                logger.warning("OpenAI not available - using mock responses")
        
        elif self.llm_provider.lower() == "huggingface":
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Use a smaller model for demo
                    self.llm_pipeline = pipeline(
                        "text-generation",
                        model="microsoft/DialoGPT-small",
                        tokenizer="microsoft/DialoGPT-small"
                    )
                    self.use_real_llm = True
                    logger.info("HuggingFace LLM initialized")
                except Exception as e:
                    logger.warning(f"Failed to load HuggingFace model: {e}")
                    self.use_real_llm = False
            else:
                self.use_real_llm = False
                logger.warning("Transformers not available - using mock responses")
        
        else:
            self.use_real_llm = False
            logger.warning(f"Unknown LLM provider: {self.llm_provider}")
    
    def load_documents(self, documents: List[Dict[str, str]]) -> None:
        """Load and process documents into the RAG system"""
        
        logger.info(f"Loading {len(documents)} documents...")
        
        all_chunks = []
        
        for doc in documents:
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            
            # Process document
            chunks = self.processor.chunk_text(content, title)
            all_chunks.extend(chunks)
            
            logger.info(f"Processed '{title}': {len(chunks)} chunks")
        
        # Add to vector database
        self.vector_db.add_documents(all_chunks)
        
        logger.info(f"Successfully loaded {len(all_chunks)} chunks total")
    
    def load_pdf(self, pdf_path: str) -> None:
        """Load PDF file into the RAG system"""
        
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Extract text from PDF
        text, metadata = self.processor.extract_text_from_pdf(pdf_path)
        
        # Create document
        doc_name = os.path.basename(pdf_path)
        chunks = self.processor.chunk_text(text, doc_name)
        
        # Add metadata to chunks
        for chunk in chunks:
            if chunk.metadata:
                chunk.metadata.update(metadata)
            else:
                chunk.metadata = metadata.copy()
        
        # Add to vector database
        self.vector_db.add_documents(chunks)
        
        logger.info(f"Successfully loaded PDF: {len(chunks)} chunks")
    
    def create_rag_prompt(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Create optimized prompt for RAG generation
        
        Key techniques:
        1. Clear instruction and role definition
        2. Structured context presentation
        3. Source attribution requirements
        4. Grounding constraints to prevent hallucination
        """
        
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n--- Source {i}: {chunk.source} (Page {chunk.page_number}) ---\n"
            context_text += chunk.content[:800] + ("..." if len(chunk.content) > 800 else "")
            context_text += "\n"
        
        prompt = f"""You are an expert research assistant. Your task is to answer questions based ONLY on the provided context documents. Follow these guidelines:

1. ONLY use information from the provided context
2. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided documents"
3. Always cite your sources by mentioning the document name and page number
4. Be comprehensive but concise
5. If there are conflicting information in different sources, mention this

CONTEXT DOCUMENTS:
{context_text}

QUESTION: {question}

ANSWER (with source citations):"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        
        if not self.use_real_llm:
            return self._mock_llm_response(prompt)
        
        try:
            if self.llm_provider.lower() == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1  # Low temperature for factual responses
                )
                return response.choices[0].message.content
            
            elif self.llm_provider.lower() == "huggingface":
                response = self.llm_pipeline(prompt, max_length=1000, num_return_sequences=1)
                return response[0]["generated_text"][len(prompt):].strip()
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._mock_llm_response(prompt)
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Generate mock response when real LLM is unavailable"""
        
        # Extract question from prompt
        question_match = re.search(r'QUESTION: (.*?)\n', prompt)
        question = question_match.group(1) if question_match else "the question"
        
        mock_responses = [
            f"Based on the provided documents, I can partially answer {question}. The context suggests that this topic involves multiple considerations including technical implementation, best practices, and potential challenges. However, for a complete answer, additional information would be needed beyond what's available in the current documents. Sources: Document analysis indicates relevant information across multiple sections.",
            
            f"According to the documentation provided, {question} can be addressed through several approaches. The sources mention key principles and methodologies that are commonly applied in this domain. The documents provide foundational information, though specific implementation details may require additional research. Sources: Multiple documents referenced in the context.",
            
            f"The provided context offers insights into {question}. From the available information, it appears that best practices emphasize systematic approaches and careful consideration of various factors. While the documents provide valuable background information, a comprehensive answer would benefit from additional sources. Sources: Referenced documentation provides relevant background."
        ]
        
        return np.random.choice(mock_responses)
    
    def ask_question(self, question: str, top_k: int = 5, 
                    source_filter: Optional[str] = None) -> RAGResponse:
        """
        Ask a question to the RAG system
        
        Process:
        1. Retrieve relevant chunks using semantic search
        2. Create context-aware prompt
        3. Generate response using LLM
        4. Return structured response with sources
        """
        
        start_time = time.time()
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Retrieve relevant chunks
        relevant_chunks, retrieval_scores = self.vector_db.search(
            question, top_k=top_k, filter_source=source_filter
        )
        
        if not relevant_chunks:
            response = RAGResponse(
                question=question,
                answer="No relevant information found in the knowledge base.",
                source_chunks=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                retrieval_scores=[]
            )
            return response
        
        # Create RAG prompt
        prompt = self.create_rag_prompt(question, relevant_chunks)
        
        # Generate response
        answer = self.generate_response(prompt)
        
        # Calculate confidence score based on retrieval scores
        confidence_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
        
        response = RAGResponse(
            question=question,
            answer=answer,
            source_chunks=relevant_chunks,
            confidence_score=float(confidence_score),
            processing_time=time.time() - start_time,
            retrieval_scores=retrieval_scores
        )
        
        # Store conversation
        self.conversation_history.append(response)
        
        logger.info(f"Generated response in {response.processing_time:.2f}s")
        
        return response
    
    def interactive_chat(self) -> None:
        """Interactive chat interface for the RAG system"""
        
        print("\n🤖 RAG CHAT SYSTEM - Chat with Your Documents!")
        print("=" * 60)
        print("Type your questions about the loaded documents.")
        print("Commands: 'quit' to exit, 'stats' for database info, 'history' for conversation history")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using RAG Chat!")
                    break
                
                elif question.lower() == 'stats':
                    stats = self.vector_db.get_statistics()
                    print("\n📊 Database Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue
                
                elif question.lower() == 'history':
                    print(f"\n📜 Conversation History ({len(self.conversation_history)} items):")
                    for i, response in enumerate(self.conversation_history[-5:], 1):
                        print(f"   {i}. Q: {response.question[:50]}...")
                        print(f"      A: {response.answer[:100]}...")
                    continue
                
                elif not question:
                    continue
                
                # Process question
                print("\n🔍 Searching knowledge base...")
                response = self.ask_question(question)
                
                # Display response
                print(f"\n🤖 Answer:")
                print(f"{response.answer}\n")
                
                print(f"📚 Sources ({len(response.source_chunks)}):")
                for i, chunk in enumerate(response.source_chunks, 1):
                    print(f"   {i}. {chunk.source} (Page {chunk.page_number}, Score: {response.retrieval_scores[i-1]:.3f})")
                
                print(f"\n⚡ Confidence: {response.confidence_score:.2%} | Time: {response.processing_time:.2f}s")
            
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
    def evaluate_system(self, test_questions: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Evaluate RAG system performance
        
        Metrics:
        - Retrieval accuracy: Are relevant chunks retrieved?
        - Answer quality: Subjective evaluation
        - Response time: Latency measurements
        - Coverage: How much of knowledge base is accessible?
        """
        
        print("\n📊 EVALUATING RAG SYSTEM PERFORMANCE")
        print("=" * 50)
        
        if not test_questions:
            # Create default test questions
            test_questions = [
                {
                    "question": "What are the key steps in data preparation for machine learning?",
                    "expected_source": "Machine Learning Best Practices Guide"
                },
                {
                    "question": "What is the attention mechanism in transformers?",
                    "expected_source": "Deep Learning Architecture Guide"
                },
                {
                    "question": "What are the main sources of bias in AI systems?",
                    "expected_source": "AI Ethics and Responsible Development"
                },
                {
                    "question": "How do CNNs process images?",
                    "expected_source": "Deep Learning Architecture Guide"
                },
                {
                    "question": "What are privacy-preserving techniques in AI?",
                    "expected_source": "AI Ethics and Responsible Development"
                }
            ]
        
        evaluation_results = {
            "total_questions": len(test_questions),
            "responses": [],
            "metrics": {
                "avg_response_time": 0,
                "avg_confidence": 0,
                "retrieval_accuracy": 0,
                "answer_coverage": 0
            }
        }
        
        response_times = []
        confidence_scores = []
        retrieval_hits = 0
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n🧪 Test {i}/{len(test_questions)}: {test['question'][:60]}...")
            
            response = self.ask_question(test["question"])
            
            # Check retrieval accuracy
            retrieved_sources = [chunk.source for chunk in response.source_chunks]
            expected_source = test.get("expected_source", "")
            
            if expected_source and any(expected_source in source for source in retrieved_sources):
                retrieval_hits += 1
                retrieval_success = True
            else:
                retrieval_success = False
            
            # Store results
            test_result = {
                "question": test["question"],
                "answer": response.answer,
                "confidence": response.confidence_score,
                "response_time": response.processing_time,
                "sources_found": len(response.source_chunks),
                "retrieval_success": retrieval_success,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources
            }
            
            evaluation_results["responses"].append(test_result)
            
            response_times.append(response.processing_time)
            confidence_scores.append(response.confidence_score)
            
            print(f"   ✅ Confidence: {response.confidence_score:.2%}")
            print(f"   ⏱️  Time: {response.processing_time:.2f}s")
            print(f"   📚 Sources: {len(response.source_chunks)}")
            print(f"   🎯 Retrieval: {'✓' if retrieval_success else '✗'}")
        
        # Calculate metrics
        evaluation_results["metrics"] = {
            "avg_response_time": np.mean(response_times),
            "avg_confidence": np.mean(confidence_scores),
            "retrieval_accuracy": retrieval_hits / len(test_questions),
            "answer_coverage": len([r for r in evaluation_results["responses"] if len(r["retrieved_sources"]) > 0]) / len(test_questions)
        }
        
        print(f"\n📈 EVALUATION RESULTS:")
        print(f"   Average Response Time: {evaluation_results['metrics']['avg_response_time']:.2f}s")
        print(f"   Average Confidence: {evaluation_results['metrics']['avg_confidence']:.2%}")
        print(f"   Retrieval Accuracy: {evaluation_results['metrics']['retrieval_accuracy']:.2%}")
        print(f"   Answer Coverage: {evaluation_results['metrics']['answer_coverage']:.2%}")
        
        # Save evaluation results
        eval_path = f"{self.output_dir}/evaluations/rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"\n💾 Evaluation results saved to: {eval_path}")
        
        return evaluation_results
    
    def visualize_system_performance(self, evaluation_results: Dict[str, Any] = None):
        """Create visualizations of RAG system performance"""
        
        if not evaluation_results:
            print("No evaluation results to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        responses = evaluation_results["responses"]
        
        # 1. Response Time Distribution
        response_times = [r["response_time"] for r in responses]
        ax1.hist(response_times, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Response Time Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(response_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(response_times):.2f}s')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence Scores
        confidence_scores = [r["confidence"] for r in responses]
        questions = [f"Q{i+1}" for i in range(len(responses))]
        
        bars = ax2.bar(questions, confidence_scores, 
                      color=['green' if c > 0.7 else 'orange' if c > 0.4 else 'red' for c in confidence_scores])
        ax2.set_title('Confidence Scores by Question', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1)
        
        for bar, score in zip(bars, confidence_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Retrieval Success Rate
        retrieval_success = [1 if r["retrieval_success"] else 0 for r in responses]
        success_rate = np.mean(retrieval_success)
        
        ax3.pie([success_rate, 1-success_rate], 
               labels=['Successful', 'Failed'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax3.set_title('Retrieval Success Rate', fontweight='bold', fontsize=14)
        
        # 4. Sources Found Distribution
        sources_found = [r["sources_found"] for r in responses]
        ax4.bar(questions, sources_found, color='lightblue')
        ax4.set_title('Number of Sources Found per Question', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Sources Found')
        ax4.set_xlabel('Questions')
        
        for i, count in enumerate(sources_found):
            ax4.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/evaluations/rag_performance_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_system_report(self) -> str:
        """Generate comprehensive system report"""
        
        db_stats = self.vector_db.get_statistics()
        
        report = f"""
RAG SYSTEM ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM CONFIGURATION
{'='*30}
📊 Vector Database: {self.vector_db.embedding_model_name}
🤖 LLM Provider: {self.llm_provider}
📝 Document Processor: Chunk size {self.processor.chunk_size}, Overlap {self.processor.overlap}
💾 Output Directory: {self.output_dir}

KNOWLEDGE BASE STATISTICS
{'='*40}
📚 Total Chunks: {db_stats.get('total_chunks', 0):,}
🗂️  Unique Sources: {db_stats.get('unique_sources', 0)}
📄 Total Words: {db_stats.get('total_words', 0):,}
📐 Embedding Dimension: {db_stats.get('embedding_dimension', 0)}
📈 Average Words/Chunk: {db_stats.get('avg_chunk_words', 0):.1f}

SOURCE DISTRIBUTION:
"""
        
        if 'source_distribution' in db_stats:
            for source, count in db_stats['source_distribution'].items():
                report += f"   • {source}: {count} chunks\n"
        
        report += f"""

CONVERSATION HISTORY
{'='*30}
💬 Total Conversations: {len(self.conversation_history)}
"""
        
        if self.conversation_history:
            avg_confidence = np.mean([r.confidence_score for r in self.conversation_history])
            avg_response_time = np.mean([r.processing_time for r in self.conversation_history])
            avg_sources = np.mean([len(r.source_chunks) for r in self.conversation_history])
            
            report += f"""📊 Average Confidence: {avg_confidence:.2%}
⏱️  Average Response Time: {avg_response_time:.2f}s
📚 Average Sources per Response: {avg_sources:.1f}

RECENT QUESTIONS:
"""
            for i, response in enumerate(self.conversation_history[-5:], 1):
                report += f"   {i}. {response.question[:80]}...\n"
                report += f"      Confidence: {response.confidence_score:.2%}, Sources: {len(response.source_chunks)}\n"
        
        report += f"""

SYSTEM CAPABILITIES
{'='*30}
✅ Document Processing: PDF, Text
✅ Semantic Search: Vector embeddings + FAISS
✅ Context-Aware Generation: RAG prompting
✅ Source Attribution: Automatic citation
✅ Performance Monitoring: Confidence scoring
✅ Interactive Chat: Real-time Q&A
✅ Evaluation Framework: Automated testing

RECOMMENDATIONS
{'='*20}
📈 Performance: Current system shows {"good" if avg_confidence > 0.6 else "moderate"} confidence levels
🔧 Optimization: Consider {"increasing chunk overlap" if avg_sources < 3 else "tuning retrieval parameters"}
📚 Content: {"Knowledge base well-populated" if db_stats.get('total_chunks', 0) > 50 else "Consider adding more documents"}
🚀 Scaling: System ready for {"production deployment" if db_stats.get('total_chunks', 0) > 100 else "additional testing"}

NEXT STEPS
{'='*15}
1. Add more domain-specific documents
2. Fine-tune embedding model if needed
3. Implement user feedback collection
4. Add multilingual support
5. Optimize for specific use cases
6. Deploy as web service
7. Integrate with existing systems
"""
        
        # Save report
        report_path = f"{self.output_dir}/rag_system_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 System report saved to: {report_path}")
        return report
    
    def run_complete_demo(self):
        """Run complete RAG system demonstration"""
        
        print("🔍 RAG SYSTEM: CHAT WITH YOUR OWN PDFS")
        print("=" * 80)
        print("This demonstration shows:")
        print("1. Document processing and intelligent chunking")
        print("2. Vector embeddings and semantic search")  
        print("3. Context-aware LLM generation")
        print("4. Source attribution and confidence scoring")
        print("5. Performance evaluation and optimization")
        print("=" * 80)
        
        try:
            # 1. Load sample documents
            print("\n📚 Loading sample documents...")
            sample_docs = self.processor.create_sample_documents()
            self.load_documents(sample_docs)
            
            # 2. Display database statistics
            print("\n📊 Knowledge Base Statistics:")
            stats = self.vector_db.get_statistics()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # 3. Run evaluation
            print("\n🧪 Running system evaluation...")
            evaluation_results = self.evaluate_system()
            
            # 4. Generate visualizations
            print("\n📈 Creating performance visualizations...")
            self.visualize_system_performance(evaluation_results)
            
            # 5. Demo some questions
            demo_questions = [
                "What are the key principles of ethical AI development?",
                "How do you prevent overfitting in machine learning models?",
                "What is the attention mechanism and why is it important?",
                "What are the main components of a CNN architecture?"
            ]
            
            print(f"\n💬 Demonstrating Q&A with sample questions...")
            for i, question in enumerate(demo_questions, 1):
                print(f"\n📝 Demo Question {i}: {question}")
                response = self.ask_question(question)
                print(f"🤖 Answer: {response.answer[:200]}...")
                print(f"📚 Sources: {len(response.source_chunks)} | Confidence: {response.confidence_score:.2%}")
            
            # 6. Generate comprehensive report
            print(f"\n📄 Generating system report...")
            self.generate_system_report()
            
            # 7. Business impact analysis
            self._analyze_business_impact()
            
            # 8. Offer interactive chat
            print(f"\n🎮 Would you like to try interactive chat? (y/n)")
            if input().lower().startswith('y'):
                self.interactive_chat()
            
            print(f"\n✅ RAG SYSTEM DEMO COMPLETE!")
            print(f"📁 All results saved to: {self.output_dir}/")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_business_impact(self):
        """Analyze business impact and ROI of RAG system"""
        
        print(f"\n💼 BUSINESS IMPACT ANALYSIS")
        print("=" * 50)
        
        # Calculate time savings
        docs_in_kb = len(self.vector_db.chunks)
        manual_search_time = docs_in_kb * 2  # 2 minutes per document manually
        rag_search_time = len(self.conversation_history) * 0.1  # 6 seconds average per query
        
        time_saved = manual_search_time - rag_search_time
        hourly_rate = 75  # Knowledge worker hourly rate
        cost_savings = (time_saved / 60) * hourly_rate
        
        print(f"⏱️  Time Analysis:")
        print(f"   Documents in KB: {docs_in_kb}")
        print(f"   Manual search time: {manual_search_time} minutes")
        print(f"   RAG search time: {rag_search_time:.1f} minutes")
        print(f"   Time saved: {time_saved:.1f} minutes")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Knowledge worker rate: ${hourly_rate}/hour")
        print(f"   Cost savings: ${cost_savings:.2f}")
        print(f"   Efficiency improvement: {((time_saved/manual_search_time)*100):.1f}%")
        
        print(f"\n🎯 Business Applications:")
        applications = [
            "Customer Support: Instant answers from product documentation",
            "Legal Research: Quick queries across case law and regulations", 
            "Medical Reference: Access to latest research and guidelines",
            "Employee Training: Interactive learning from company materials",
            "Sales Support: Product information retrieval during calls"
        ]
        
        for app in applications:
            print(f"   • {app}")
        
        print(f"\n📈 Scalability Benefits:")
        print("   • 24/7 availability without human intervention")
        print("   • Consistent quality regardless of query volume")
        print("   • Easy knowledge base updates and versioning")
        print("   • Multi-language support with proper models")
        print("   • Integration with existing enterprise systems")

if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(llm_provider="openai", embedding_model="all-MiniLM-L6-v2")
    
    # Run complete demonstration
    rag.run_complete_demo()
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNING OUTCOMES:")
    print("1. RAG combines retrieval and generation for knowledge-intensive tasks")
    print("2. Document chunking strategies significantly impact retrieval quality")
    print("3. Vector embeddings enable semantic search beyond keyword matching")
    print("4. Context management is crucial for accurate LLM responses")
    print("5. Source attribution builds trust and enables verification")
    print("\n🚀 NEXT STEPS:")
    print("• Experiment with different embedding models")
    print("• Optimize chunk size and overlap for your documents")  
    print("• Implement user feedback loops for continuous improvement")
    print("• Add support for more document formats")
    print("• Deploy as a web service for team access")
    print("="*80)