#!/usr/bin/env python3
"""
Monitoring & Drift Detection System
Project 19 - PHASE 05: MLOps & Deployment

A comprehensive system to monitor ML models and detect data/concept drift.
Author: Your Name
Date: December 2024
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import json
import warnings
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Class for detecting data and concept drift"""
    
    def __init__(self, reference_data: pd.DataFrame):
        """Initialize with reference data"""
        self.reference_data = reference_data
        self.reference_stats = self._calculate_reference_stats()
        
    def _calculate_reference_stats(self) -> Dict:
        """Calculate reference statistics"""
        stats_dict = {}
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['float64', 'int64']:
                stats_dict[col] = {
                    'mean': self.reference_data[col].mean(),
                    'std': self.reference_data[col].std(),
                    'min': self.reference_data[col].min(),
                    'max': self.reference_data[col].max(),
                    'quantiles': self.reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
        return stats_dict
    
    def calculate_psi(self, current_data: pd.DataFrame, feature: str, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            # Define bins based on reference data
            _, bin_edges = np.histogram(ref_values, bins=bins)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(ref_values, bins=bin_edges)
            cur_hist, _ = np.histogram(cur_values, bins=bin_edges)
            
            # Normalize to get proportions
            ref_prop = ref_hist / ref_hist.sum()
            cur_prop = cur_hist / cur_hist.sum()
            
            # Add small constant to avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)
            
            # Calculate PSI
            psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
            
            return psi
            
        except Exception as e:
            logger.error(f"PSI calculation failed for {feature}: {str(e)}")
            return np.nan
    
    def kolmogorov_smirnov_test(self, current_data: pd.DataFrame, feature: str) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test"""
        try:
            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            statistic, p_value = ks_2samp(ref_values, cur_values)
            return statistic, p_value
            
        except Exception as e:
            logger.error(f"KS test failed for {feature}: {str(e)}")
            return np.nan, np.nan
    
    def wasserstein_distance(self, current_data: pd.DataFrame, feature: str) -> float:
        """Calculate Wasserstein distance"""
        try:
            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            distance = stats.wasserstein_distance(ref_values, cur_values)
            return distance
            
        except Exception as e:
            logger.error(f"Wasserstein distance calculation failed for {feature}: {str(e)}")
            return np.nan
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data drift detection"""
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': current_data.shape,
            'features_analyzed': [],
            'drift_scores': {},
            'drift_detected': False,
            'alerts': []
        }
        
        # Define thresholds
        psi_threshold = 0.2
        ks_threshold = 0.05
        
        for feature in current_data.columns:
            if current_data[feature].dtype in ['float64', 'int64']:
                feature_results = {}
                
                # PSI
                psi_score = self.calculate_psi(current_data, feature)
                feature_results['psi'] = psi_score
                
                # KS test
                ks_stat, ks_p_value = self.kolmogorov_smirnov_test(current_data, feature)
                feature_results['ks_statistic'] = ks_stat
                feature_results['ks_p_value'] = ks_p_value
                
                # Wasserstein distance
                ws_distance = self.wasserstein_distance(current_data, feature)
                feature_results['wasserstein_distance'] = ws_distance
                
                # Determine drift status
                drift_indicators = []
                if psi_score > psi_threshold:
                    drift_indicators.append('PSI')
                if ks_p_value < ks_threshold:
                    drift_indicators.append('KS_TEST')
                
                feature_results['drift_indicators'] = drift_indicators
                feature_results['drift_detected'] = len(drift_indicators) > 0
                
                if feature_results['drift_detected']:
                    drift_results['drift_detected'] = True
                    drift_results['alerts'].append(f"Drift detected in {feature}: {', '.join(drift_indicators)}")
                
                drift_results['drift_scores'][feature] = feature_results
                drift_results['features_analyzed'].append(feature)
        
        return drift_results

class ModelPerformanceMonitor:
    """Monitor model performance and detect concept drift"""
    
    def __init__(self, model, reference_performance: Dict[str, float]):
        """Initialize with model and reference performance"""
        self.model = model
        self.reference_performance = reference_performance
        self.performance_history = []
        
    def calculate_performance_metrics(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            y_pred = self.model.predict(X)
            
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {str(e)}")
            return {}
    
    def detect_performance_drift(self, X: pd.DataFrame, y_true: pd.Series, 
                               threshold: float = 0.05) -> Dict[str, Any]:
        """Detect concept drift based on performance degradation"""
        current_metrics = self.calculate_performance_metrics(X, y_true)
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'current_performance': current_metrics,
            'reference_performance': self.reference_performance,
            'performance_changes': {},
            'drift_detected': False,
            'alerts': []
        }
        
        for metric, current_value in current_metrics.items():
            if metric in self.reference_performance:
                reference_value = self.reference_performance[metric]
                
                # Calculate performance change
                if metric in ['mae', 'mse', 'rmse']:  # Lower is better
                    change = (current_value - reference_value) / reference_value
                    drift_condition = change > threshold
                else:  # Higher is better (e.g., r2)
                    change = (reference_value - current_value) / reference_value
                    drift_condition = change > threshold
                
                drift_results['performance_changes'][metric] = {
                    'current': current_value,
                    'reference': reference_value,
                    'change_pct': change * 100,
                    'drift_detected': drift_condition
                }
                
                if drift_condition:
                    drift_results['drift_detected'] = True
                    drift_results['alerts'].append(
                        f"Performance drift in {metric}: {change*100:.2f}% degradation"
                    )
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics
        })
        
        return drift_results

class DataQualityMonitor:
    """Monitor data quality metrics"""
    
    def __init__(self, reference_data: pd.DataFrame):
        """Initialize with reference data quality metrics"""
        self.reference_data = reference_data
        self.reference_quality = self._calculate_reference_quality()
        
    def _calculate_reference_quality(self) -> Dict[str, Any]:
        """Calculate reference data quality metrics"""
        quality_metrics = {
            'missing_percentage': self.reference_data.isnull().sum() / len(self.reference_data),
            'duplicate_percentage': self.reference_data.duplicated().sum() / len(self.reference_data),
            'zero_percentage': (self.reference_data == 0).sum() / len(self.reference_data)
        }
        return quality_metrics
    
    def assess_data_quality(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current data quality"""
        current_quality = {
            'missing_percentage': current_data.isnull().sum() / len(current_data),
            'duplicate_percentage': current_data.duplicated().sum() / len(current_data),
            'zero_percentage': (current_data == 0).sum() / len(current_data)
        }
        
        quality_results = {
            'timestamp': datetime.now().isoformat(),
            'current_quality': current_quality.to_dict(),
            'quality_changes': {},
            'quality_issues': [],
            'overall_score': 100.0
        }
        
        # Compare with reference
        for metric in current_quality.index:
            if metric in self.reference_quality['missing_percentage'].index:
                current_val = current_quality[metric].mean()
                reference_val = self.reference_quality['missing_percentage'][metric]
                change = current_val - reference_val
                
                quality_results['quality_changes'][metric] = {
                    'current': current_val,
                    'reference': reference_val,
                    'change': change
                }
                
                # Define quality thresholds
                if change > 0.05:  # 5% threshold
                    quality_results['quality_issues'].append(f"Quality degradation in {metric}")
                    quality_results['overall_score'] -= 10
        
        return quality_results

class MonitoringDashboard:
    """Create monitoring dashboard with Streamlit"""
    
    def __init__(self, drift_detector: DriftDetector, performance_monitor: ModelPerformanceMonitor):
        self.drift_detector = drift_detector
        self.performance_monitor = performance_monitor
        
    def create_drift_visualization(self, drift_results: Dict[str, Any]) -> go.Figure:
        """Create drift visualization"""
        features = list(drift_results['drift_scores'].keys())
        psi_scores = [drift_results['drift_scores'][f]['psi'] for f in features]
        ks_stats = [drift_results['drift_scores'][f]['ks_statistic'] for f in features]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['PSI Scores', 'KS Statistics'],
            vertical_spacing=0.1
        )
        
        # PSI scores
        fig.add_trace(
            go.Bar(x=features, y=psi_scores, name='PSI Score', 
                   marker_color=['red' if p > 0.2 else 'blue' for p in psi_scores]),
            row=1, col=1
        )
        
        # Add PSI threshold line
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=1, col=1)
        
        # KS statistics
        fig.add_trace(
            go.Bar(x=features, y=ks_stats, name='KS Statistic',
                   marker_color=['red' if k > 0.1 else 'blue' for k in ks_stats]),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Data Drift Detection Results',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_performance_trend(self) -> go.Figure:
        """Create performance trend visualization"""
        if not self.performance_monitor.performance_history:
            return go.Figure()
        
        timestamps = [h['timestamp'] for h in self.performance_monitor.performance_history]
        r2_scores = [h['metrics']['r2'] for h in self.performance_monitor.performance_history]
        mae_scores = [h['metrics']['mae'] for h in self.performance_monitor.performance_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['R2 Score Trend', 'MAE Trend'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=r2_scores, mode='lines+markers', name='R2 Score'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=mae_scores, mode='lines+markers', name='MAE'),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Model Performance Trends',
            height=600,
            showlegend=False
        )
        
        return fig

class MLMonitoringSystem:
    """Complete ML monitoring system"""
    
    def __init__(self):
        """Initialize monitoring system"""
        self.setup_logging()
        self.setup_directories()
        self.drift_detector = None
        self.performance_monitor = None
        self.quality_monitor = None
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('outputs/logs', exist_ok=True)
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'outputs/reports', 'outputs/dashboards', 
            'outputs/metrics', 'outputs/logs',
            'data/reference', 'data/production'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_monitoring(self, reference_data: pd.DataFrame, model, 
                            reference_performance: Dict[str, float]):
        """Initialize monitoring components"""
        self.drift_detector = DriftDetector(reference_data)
        self.performance_monitor = ModelPerformanceMonitor(model, reference_performance)
        self.quality_monitor = DataQualityMonitor(reference_data)
        
        logger.info("Monitoring system initialized successfully")
    
    def monitor_batch(self, current_data: pd.DataFrame, y_true: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Monitor a batch of new data"""
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(current_data),
            'monitoring_status': 'SUCCESS'
        }
        
        try:
            # Data drift detection
            if self.drift_detector:
                drift_results = self.drift_detector.detect_data_drift(current_data)
                monitoring_results['data_drift'] = drift_results
                
            # Performance monitoring (if labels available)
            if self.performance_monitor and y_true is not None:
                performance_results = self.performance_monitor.detect_performance_drift(
                    current_data, y_true
                )
                monitoring_results['performance_drift'] = performance_results
                
            # Data quality assessment
            if self.quality_monitor:
                quality_results = self.quality_monitor.assess_data_quality(current_data)
                monitoring_results['data_quality'] = quality_results
            
            # Generate summary alerts
            monitoring_results['alerts'] = self._generate_summary_alerts(monitoring_results)
            
            # Save results
            self._save_monitoring_results(monitoring_results)
            
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            monitoring_results['monitoring_status'] = 'FAILED'
            monitoring_results['error'] = str(e)
        
        return monitoring_results
    
    def _generate_summary_alerts(self, results: Dict[str, Any]) -> List[str]:
        """Generate summary of all alerts"""
        all_alerts = []
        
        if 'data_drift' in results:
            all_alerts.extend(results['data_drift'].get('alerts', []))
            
        if 'performance_drift' in results:
            all_alerts.extend(results['performance_drift'].get('alerts', []))
            
        if 'data_quality' in results:
            all_alerts.extend(results['data_quality'].get('quality_issues', []))
        
        return all_alerts
    
    def _save_monitoring_results(self, results: Dict[str, Any]):
        """Save monitoring results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/reports/monitoring_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Monitoring results saved to {filename}")

def create_sample_drift_scenario():
    """Create sample data with artificial drift for demonstration"""
    # Load original data
    housing = fetch_california_housing(as_frame=True)
    original_data = housing.frame
    
    # Create reference data (first 80%)
    reference_data = original_data.iloc[:int(len(original_data) * 0.8)]
    
    # Create drifted data (modify some features)
    drifted_data = original_data.iloc[int(len(original_data) * 0.8):].copy()
    
    # Introduce artificial drift
    # Shift median income distribution
    drifted_data['MedInc'] = drifted_data['MedInc'] * 1.5 + np.random.normal(0, 0.1, len(drifted_data))
    
    # Shift house age distribution
    drifted_data['HouseAge'] = drifted_data['HouseAge'] * 0.8 + np.random.normal(0, 2, len(drifted_data))
    
    return reference_data, drifted_data

def main():
    """Main function to demonstrate monitoring system"""
    logger.info("Starting ML Monitoring System Demo")
    
    try:
        # Create sample data with drift
        reference_data, current_data = create_sample_drift_scenario()
        
        # Prepare data for modeling
        X_ref = reference_data.drop('MedHouseVal', axis=1)
        y_ref = reference_data['MedHouseVal']
        X_current = current_data.drop('MedHouseVal', axis=1)
        y_current = current_data['MedHouseVal']
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_ref, y_ref)
        
        # Calculate reference performance
        y_pred_ref = model.predict(X_ref)
        reference_performance = {
            'mae': mean_absolute_error(y_ref, y_pred_ref),
            'mse': mean_squared_error(y_ref, y_pred_ref),
            'rmse': np.sqrt(mean_squared_error(y_ref, y_pred_ref)),
            'r2': r2_score(y_ref, y_pred_ref)
        }
        
        # Initialize monitoring system
        monitoring_system = MLMonitoringSystem()
        monitoring_system.initialize_monitoring(X_ref, model, reference_performance)
        
        # Monitor current batch
        results = monitoring_system.monitor_batch(X_current, y_current)
        
        # Display results
        print("\n" + "="*60)
        print("ML MONITORING SYSTEM RESULTS")
        print("="*60)
        
        print(f"Timestamp: {results['timestamp']}")
        print(f"Batch Size: {results['batch_size']}")
        print(f"Status: {results['monitoring_status']}")
        
        if 'data_drift' in results:
            drift_detected = results['data_drift']['drift_detected']
            print(f"\nData Drift Detected: {drift_detected}")
            if drift_detected:
                print("Drift Alerts:")
                for alert in results['data_drift']['alerts']:
                    print(f"  - {alert}")
        
        if 'performance_drift' in results:
            perf_drift = results['performance_drift']['drift_detected']
            print(f"\nPerformance Drift Detected: {perf_drift}")
            if perf_drift:
                print("Performance Alerts:")
                for alert in results['performance_drift']['alerts']:
                    print(f"  - {alert}")
        
        print(f"\nTotal Alerts: {len(results['alerts'])}")
        for alert in results['alerts']:
            print(f"  - {alert}")
        
        print("\n" + "="*60)
        print("Results saved to outputs/reports/")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()