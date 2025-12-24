#!/usr/bin/env python3
"""
End-to-End ML Pipeline
Project 18 - PHASE 05: MLOps & Deployment

A complete automated ML pipeline with data ingestion, training, and deployment.
Author: Your Name
Date: December 2024
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Complete ML Pipeline with data ingestion, training, and deployment"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.setup_mlflow()
        self.setup_directories()
        
        # Pipeline state
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_selector = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration"""
        return {
            'data_ingestion': {
                'dataset_name': 'california_housing',
                'test_size': 0.2,
                'random_state': 42
            },
            'feature_engineering': {
                'scaling': True,
                'feature_selection': True,
                'k_best_features': 5
            },
            'model_training': {
                'algorithms': ['linear_regression', 'random_forest'],
                'cross_validation': True,
                'cv_folds': 5
            },
            'evaluation': {
                'metrics': ['mae', 'mse', 'rmse', 'r2'],
                'threshold_r2': 0.5
            }
        }
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_experiment("ml_pipeline_experiment")
        logger.info("MLflow experiment set up")
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['outputs/data', 'outputs/models', 'outputs/reports', 'outputs/logs']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        logger.info("Directory structure created")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        logger.info("Starting ML Pipeline execution")
        
        pipeline_start = datetime.now()
        results = {}
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            self.ingest_data()
            results['data_ingestion'] = 'SUCCESS'
            
            # Step 2: Data Validation
            logger.info("Step 2: Data Validation")
            validation_results = self.validate_data()
            results['data_validation'] = validation_results
            
            # Step 3: Feature Engineering
            logger.info("Step 3: Feature Engineering")
            self.engineer_features()
            results['feature_engineering'] = 'SUCCESS'
            
            # Step 4: Model Training
            logger.info("Step 4: Model Training")
            training_results = self.train_models()
            results['model_training'] = training_results
            
            # Step 5: Model Evaluation
            logger.info("Step 5: Model Evaluation")
            evaluation_results = self.evaluate_models()
            results['model_evaluation'] = evaluation_results
            
            # Step 6: Model Selection
            logger.info("Step 6: Model Selection")
            selection_results = self.select_best_model()
            results['model_selection'] = selection_results
            
            # Step 7: Model Deployment
            logger.info("Step 7: Model Deployment")
            deployment_results = self.deploy_model()
            results['model_deployment'] = deployment_results
            
            # Calculate total execution time
            total_time = (datetime.now() - pipeline_start).total_seconds()
            results['total_execution_time'] = total_time
            results['pipeline_status'] = 'SUCCESS'
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results['pipeline_status'] = 'FAILED'
            results['error'] = str(e)
            raise
        
        # Save pipeline results
        self.save_pipeline_results(results)
        
        return results
    
    def ingest_data(self):
        """Ingest data from the specified source"""
        try:
            # For this example, we'll use California housing dataset
            # In production, this would connect to databases, APIs, etc.
            data = fetch_california_housing(as_frame=True)
            
            # Create DataFrame with features and target
            df = data.frame
            
            # Save raw data
            df.to_csv('outputs/data/raw_data.csv', index=False)
            self.raw_data = df
            
            logger.info(f"Data ingested successfully. Shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate data quality and schema"""
        if self.raw_data is None:
            raise ValueError("No data to validate")
        
        validation_results = {}
        
        try:
            # Basic data quality checks
            validation_results['total_rows'] = len(self.raw_data)
            validation_results['total_columns'] = len(self.raw_data.columns)
            validation_results['missing_values'] = self.raw_data.isnull().sum().sum()
            validation_results['duplicate_rows'] = self.raw_data.duplicated().sum()
            
            # Check for infinite values
            numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
            validation_results['infinite_values'] = np.isinf(self.raw_data[numeric_cols]).sum().sum()
            
            # Data type validation
            validation_results['data_types'] = self.raw_data.dtypes.to_dict()
            
            # Statistical validation
            validation_results['statistics'] = self.raw_data.describe().to_dict()
            
            # Quality score (simple scoring)
            quality_score = 100
            if validation_results['missing_values'] > 0:
                quality_score -= 20
            if validation_results['duplicate_rows'] > 0:
                quality_score -= 10
            if validation_results['infinite_values'] > 0:
                quality_score -= 10
            
            validation_results['quality_score'] = quality_score
            validation_results['validation_status'] = 'PASSED' if quality_score >= 70 else 'FAILED'
            
            logger.info(f"Data validation completed. Quality score: {quality_score}")
            
            # Save validation report
            pd.DataFrame([validation_results]).to_csv('outputs/reports/data_validation.csv', index=False)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def engineer_features(self):
        """Perform feature engineering"""
        if self.raw_data is None:
            raise ValueError("No data available for feature engineering")
        
        try:
            # Separate features and target
            X = self.raw_data.drop('MedHouseVal', axis=1)
            y = self.raw_data['MedHouseVal']
            
            # Split data
            test_size = self.config['data_ingestion']['test_size']
            random_state = self.config['data_ingestion']['random_state']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Feature scaling
            if self.config['feature_engineering']['scaling']:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Convert back to DataFrame
                X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
            
            # Feature selection
            if self.config['feature_engineering']['feature_selection']:
                k_features = self.config['feature_engineering']['k_best_features']
                self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
                
                X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
                X_test_selected = self.feature_selector.transform(X_test)
                
                # Get selected feature names
                selected_features = X.columns[self.feature_selector.get_support()]
                
                # Convert back to DataFrame
                X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
                X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
                
                logger.info(f"Selected features: {list(selected_features)}")
            
            # Store processed data
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            # Save processed data
            X_train.to_csv('outputs/data/X_train.csv', index=False)
            X_test.to_csv('outputs/data/X_test.csv', index=False)
            y_train.to_csv('outputs/data/y_train.csv', index=False)
            y_test.to_csv('outputs/data/y_test.csv', index=False)
            
            logger.info("Feature engineering completed successfully")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def train_models(self) -> Dict[str, Any]:
        """Train multiple models"""
        if self.X_train is None:
            raise ValueError("No training data available")
        
        training_results = {}
        
        try:
            algorithms = self.config['model_training']['algorithms']
            
            for algorithm in algorithms:
                logger.info(f"Training {algorithm}")
                
                # Get model instance
                model = self._get_model_instance(algorithm)
                
                # Train model
                with mlflow.start_run(run_name=f"{algorithm}_training"):
                    model.fit(self.X_train, self.y_train)
                    
                    # Cross-validation if enabled
                    if self.config['model_training']['cross_validation']:
                        cv_folds = self.config['model_training']['cv_folds']
                        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                                  cv=cv_folds, scoring='r2')
                        training_results[f"{algorithm}_cv_mean"] = cv_scores.mean()
                        training_results[f"{algorithm}_cv_std"] = cv_scores.std()
                        
                        # Log to MLflow
                        mlflow.log_metric("cv_r2_mean", cv_scores.mean())
                        mlflow.log_metric("cv_r2_std", cv_scores.std())
                    
                    # Store model
                    self.models[algorithm] = model
                    
                    # Save model
                    model_path = f'outputs/models/{algorithm}_model.pkl'
                    joblib.dump(model, model_path)
                    
                    # Log model to MLflow
                    mlflow.sklearn.log_model(model, f"{algorithm}_model")
                    mlflow.log_param("algorithm", algorithm)
                    
                    logger.info(f"{algorithm} training completed")
            
            training_results['models_trained'] = len(algorithms)
            training_results['training_status'] = 'SUCCESS'
            
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _get_model_instance(self, algorithm: str):
        """Get model instance based on algorithm name"""
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        if algorithm not in models:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return models[algorithm]
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all trained models"""
        if not self.models or self.X_test is None:
            raise ValueError("No models or test data available")
        
        evaluation_results = {}
        
        try:
            for algorithm, model in self.models.items():
                logger.info(f"Evaluating {algorithm}")
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(self.y_test, y_pred),
                    'mse': mean_squared_error(self.y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'r2': r2_score(self.y_test, y_pred)
                }
                
                evaluation_results[algorithm] = metrics
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"{algorithm}_evaluation"):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                
                logger.info(f"{algorithm} evaluation completed. R2: {metrics['r2']:.4f}")
            
            # Save evaluation results
            eval_df = pd.DataFrame(evaluation_results).T
            eval_df.to_csv('outputs/reports/model_evaluation.csv')
            
            evaluation_results['evaluation_status'] = 'SUCCESS'
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def select_best_model(self) -> Dict[str, Any]:
        """Select the best performing model"""
        if not self.models:
            raise ValueError("No models available for selection")
        
        try:
            # Read evaluation results
            eval_df = pd.read_csv('outputs/reports/model_evaluation.csv', index_col=0)
            
            # Select best model based on R2 score
            best_algorithm = eval_df['r2'].idxmax()
            best_r2 = eval_df.loc[best_algorithm, 'r2']
            
            # Check if model meets minimum threshold
            threshold_r2 = self.config['evaluation']['threshold_r2']
            
            if best_r2 < threshold_r2:
                logger.warning(f"Best model R2 ({best_r2:.4f}) below threshold ({threshold_r2})")
            
            self.best_model = {
                'algorithm': best_algorithm,
                'model': self.models[best_algorithm],
                'r2_score': best_r2
            }
            
            selection_results = {
                'best_algorithm': best_algorithm,
                'best_r2_score': best_r2,
                'meets_threshold': best_r2 >= threshold_r2,
                'selection_status': 'SUCCESS'
            }
            
            logger.info(f"Best model selected: {best_algorithm} (R2: {best_r2:.4f})")
            
            return selection_results
            
        except Exception as e:
            logger.error(f"Model selection failed: {str(e)}")
            raise
    
    def deploy_model(self) -> Dict[str, Any]:
        """Deploy the best model"""
        if not self.best_model:
            raise ValueError("No model selected for deployment")
        
        try:
            # Save best model with additional metadata
            deployment_package = {
                'model': self.best_model['model'],
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'algorithm': self.best_model['algorithm'],
                'r2_score': self.best_model['r2_score'],
                'deployment_date': datetime.now().isoformat(),
                'feature_names': list(self.X_train.columns)
            }
            
            # Save deployment package
            deployment_path = 'outputs/models/deployed_model.pkl'
            joblib.dump(deployment_package, deployment_path)
            
            # Register model in MLflow
            model_uri = f"outputs/models/{self.best_model['algorithm']}_model.pkl"
            model_name = "california_housing_model"
            
            with mlflow.start_run(run_name="model_deployment"):
                mlflow.sklearn.log_model(
                    self.best_model['model'], 
                    "deployed_model",
                    registered_model_name=model_name
                )
                mlflow.log_artifact(deployment_path)
            
            deployment_results = {
                'deployed_algorithm': self.best_model['algorithm'],
                'deployment_path': deployment_path,
                'model_name': model_name,
                'deployment_date': deployment_package['deployment_date'],
                'deployment_status': 'SUCCESS'
            }
            
            logger.info(f"Model deployed successfully: {self.best_model['algorithm']}")
            
            return deployment_results
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            raise
    
    def save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results"""
        try:
            # Save results as JSON and CSV
            import json
            
            with open('outputs/reports/pipeline_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Create summary DataFrame
            summary_data = {
                'Pipeline Execution Date': datetime.now().isoformat(),
                'Pipeline Status': results['pipeline_status'],
                'Data Quality Score': results.get('data_validation', {}).get('quality_score', 'N/A'),
                'Best Model': results.get('model_selection', {}).get('best_algorithm', 'N/A'),
                'Best R2 Score': results.get('model_selection', {}).get('best_r2_score', 'N/A'),
                'Total Execution Time (s)': results.get('total_execution_time', 'N/A')
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv('outputs/reports/pipeline_summary.csv', index=False)
            
            logger.info("Pipeline results saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {str(e)}")

def main():
    """Main function to run the ML pipeline"""
    try:
        # Create and run pipeline
        pipeline = MLPipeline()
        results = pipeline.run_pipeline()
        
        print("\n" + "="*50)
        print("ML PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Status: {results['pipeline_status']}")
        print(f"Total Execution Time: {results.get('total_execution_time', 'N/A'):.2f} seconds")
        
        if 'model_selection' in results:
            print(f"Best Model: {results['model_selection']['best_algorithm']}")
            print(f"Best R2 Score: {results['model_selection']['best_r2_score']:.4f}")
        
        print("\nDetailed results saved in outputs/reports/")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()