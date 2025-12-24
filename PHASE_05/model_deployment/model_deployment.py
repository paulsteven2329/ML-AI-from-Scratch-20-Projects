#!/usr/bin/env python3
"""
ML Model Deployment with FastAPI + Docker
Project 17 - PHASE 05: MLOps & Deployment

A production-ready ML model API with FastAPI and Docker containerization.
Author: Your Name
Date: December 2024
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "models/iris_classifier.pkl"
MODEL_VERSION = "1.0.0"

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    features: List[float] = Field(..., min_items=4, max_items=4, 
                                  description="4 iris features: sepal_length, sepal_width, petal_length, petal_width")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    samples: List[List[float]] = Field(..., description="List of feature arrays")
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 3.4, 5.4, 2.3],
                    [5.0, 3.0, 1.6, 0.2]
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    probability: float
    all_probabilities: Dict[str, float]
    model_version: str
    timestamp: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    batch_size: int
    total_processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    features: List[str]
    classes: List[str]
    accuracy: float
    created_at: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

class MLModel:
    """ML Model wrapper for loading and predictions"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        self.class_names = ["setosa", "versicolor", "virginica"]
        self.accuracy = 0.0
        self.created_at = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.accuracy = model_data.get('accuracy', 0.0)
                self.created_at = model_data.get('created_at', 'Unknown')
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Training new model...")
                self.train_and_save_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load model")
    
    def train_and_save_model(self):
        """Train and save a new model"""
        try:
            # Load iris dataset
            iris = load_iris()
            X, y = iris.data, iris.target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model with metadata
            model_data = {
                'model': model,
                'accuracy': accuracy,
                'created_at': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'class_names': self.class_names
            }
            
            joblib.dump(model_data, self.model_path)
            
            self.model = model
            self.accuracy = accuracy
            self.created_at = model_data['created_at']
            
            logger.info(f"Model trained and saved. Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to train model")
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make a single prediction"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Validate input
            if len(features) != 4:
                raise ValueError("Expected 4 features")
            
            # Convert to numpy array and reshape
            X = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Format response
            result = {
                'prediction': self.class_names[prediction],
                'probability': float(max(probabilities)),
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, samples: List[List[float]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            results = []
            for sample in samples:
                result = self.predict(sample)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

# Initialize the ML model
ml_model = MLModel(MODEL_PATH)

# FastAPI app initialization
app = FastAPI(
    title="ML Model API",
    description="Production-ready ML model deployment with FastAPI and Docker",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the model
async def get_model() -> MLModel:
    """Dependency to get the ML model instance"""
    return ml_model

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(model: MLModel = Depends(get_model)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model.model is not None,
        version=MODEL_VERSION
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(model: MLModel = Depends(get_model)):
    """Get model information"""
    return ModelInfo(
        model_type="RandomForestClassifier",
        model_version=MODEL_VERSION,
        features=model.feature_names,
        classes=model.class_names,
        accuracy=model.accuracy,
        created_at=model.created_at or "Unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model: MLModel = Depends(get_model)
):
    """Make a single prediction"""
    start_time = datetime.now()
    
    try:
        # Make prediction
        result = model.predict(request.features)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction, 
            request.features, 
            result['prediction'], 
            processing_time
        )
        
        return PredictionResponse(
            **result,
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model: MLModel = Depends(get_model)
):
    """Make batch predictions"""
    start_time = datetime.now()
    
    try:
        # Validate batch size
        if len(request.samples) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail="Batch size too large. Maximum 100 samples."
            )
        
        # Make predictions
        results = model.predict_batch(request.samples)
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format responses
        predictions = []
        for i, result in enumerate(results):
            pred_response = PredictionResponse(
                **result,
                model_version=MODEL_VERSION,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=total_processing_time / len(results)  # Average per prediction
            )
            predictions.append(pred_response)
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction, 
            len(request.samples), 
            total_processing_time
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.samples),
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def log_prediction(features: List[float], prediction: str, processing_time: float):
    """Background task to log predictions"""
    logger.info(
        f"Prediction logged - Features: {features}, "
        f"Prediction: {prediction}, Processing time: {processing_time:.2f}ms"
    )

async def log_batch_prediction(batch_size: int, processing_time: float):
    """Background task to log batch predictions"""
    logger.info(
        f"Batch prediction logged - Size: {batch_size}, "
        f"Total processing time: {processing_time:.2f}ms"
    )

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "model_deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )