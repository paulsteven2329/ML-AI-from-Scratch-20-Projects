"""
Loan Default Prediction using Random Forest & XGBoost
Demonstrates ensemble learning concepts: Bagging vs Boosting with feature importance analysis

Author: Your Name
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    """
    Ensemble learning pipeline for loan default prediction comparing
    Random Forest (Bagging) vs XGBoost (Boosting)
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_synthetic_loan_data(self, n_samples=10000):
        """
        Generate realistic synthetic loan dataset for demonstration
        """
        print("Generating synthetic loan dataset...")
        
        np.random.seed(42)
        
        # Generate features
        data = {}
        
        # Numerical features
        data['loan_amount'] = np.random.lognormal(10, 1, n_samples)
        data['annual_income'] = np.random.lognormal(11, 0.8, n_samples)
        data['credit_score'] = np.random.normal(650, 120, n_samples)
        data['credit_score'] = np.clip(data['credit_score'], 300, 850)
        data['employment_length'] = np.random.exponential(5, n_samples)
        data['debt_to_income'] = data['loan_amount'] / data['annual_income'] * 12
        data['loan_term'] = np.random.choice([12, 24, 36, 48, 60], n_samples, 
                                           p=[0.1, 0.2, 0.4, 0.2, 0.1])
        data['interest_rate'] = np.random.normal(12, 4, n_samples)
        data['interest_rate'] = np.clip(data['interest_rate'], 3, 25)
        
        # Categorical features
        data['loan_purpose'] = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement',
                                               'major_purchase', 'small_business', 'car', 'medical'],
                                              n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
        
        data['home_ownership'] = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], 
                                                n_samples, p=[0.4, 0.3, 0.3])
        
        data['verification_status'] = np.random.choice(['Verified', 'Source Verified', 'Not Verified'],
                                                      n_samples, p=[0.4, 0.3, 0.3])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate derived features
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        df['monthly_payment'] = df['loan_amount'] * (df['interest_rate']/100/12) / \
                               (1 - (1 + df['interest_rate']/100/12) ** (-df['loan_term']))
        df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / df['annual_income']
        
        # Generate target variable with realistic relationships
        default_probability = (
            0.1 +  # Base rate
            np.where(df['credit_score'] < 580, 0.3, 0) +  # Poor credit
            np.where(df['debt_to_income'] > 0.4, 0.2, 0) +  # High DTI
            np.where(df['loan_to_income_ratio'] > 0.5, 0.15, 0) +  # High loan ratio
            np.where(df['employment_length'] < 1, 0.1, 0) +  # Short employment
            np.where(df['verification_status'] == 'Not Verified', 0.05, 0) +
            np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        default_probability = np.clip(default_probability, 0, 1)
        df['default'] = np.random.binomial(1, default_probability)
        
        print(f"Dataset created with {len(df)} samples")
        print(f"Default rate: {df['default'].mean():.2%}")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the loan dataset
        """
        print("Preprocessing data...")
        
        # Encode categorical variables
        categorical_cols = ['loan_purpose', 'home_ownership', 'verification_status']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Select features for modeling
        feature_cols = [
            'loan_amount', 'annual_income', 'credit_score', 'employment_length',
            'debt_to_income', 'loan_term', 'interest_rate', 'loan_to_income_ratio',
            'monthly_payment', 'payment_to_income_ratio', 'loan_purpose_encoded',
            'home_ownership_encoded', 'verification_status_encoded'
        ]
        
        X = df[feature_cols].copy()
        y = df['default'].copy()
        
        # Handle any missing values
        X.fillna(X.mean(), inplace=True)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {len(X.columns)}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model (Bagging ensemble)
        """
        print("\n=== Training Random Forest (Bagging) ===")
        
        # Random Forest with optimized parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train the model
        self.rf_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, X_train, y_train, 
                                  cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                  scoring='roc_auc')
        
        print(f"Random Forest CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.rf_model
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost model (Boosting ensemble)
        """
        print("\n=== Training XGBoost (Boosting) ===")
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # XGBoost with optimized parameters
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # Train the model
        self.xgb_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.xgb_model, X_train, y_train,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                  scoring='roc_auc')
        
        print(f"XGBoost CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.xgb_model
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models and compare performance
        """
        print("\n=== Model Evaluation ===")
        
        results = {}
        
        for name, model in [("Random Forest", self.rf_model), ("XGBoost", self.xgb_model)]:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            print(results[name]['classification_report'])
        
        return results
    
    def analyze_feature_importance(self):
        """
        Analyze and compare feature importance between models
        """
        print("\n=== Feature Importance Analysis ===")
        
        # Get feature importances
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_,
            'model': 'Random Forest'
        }).sort_values('importance', ascending=False)
        
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_,
            'model': 'XGBoost'
        }).sort_values('importance', ascending=False)
        
        # Combine for comparison
        importance_df = pd.concat([rf_importance, xgb_importance], ignore_index=True)
        
        # Save to CSV
        importance_df.to_csv(f'{self.output_dir}/feature_importance_comparison.csv', index=False)
        
        print("Top 10 Features - Random Forest:")
        print(rf_importance.head(10)[['feature', 'importance']].to_string(index=False))
        
        print("\nTop 10 Features - XGBoost:")
        print(xgb_importance.head(10)[['feature', 'importance']].to_string(index=False))
        
        return importance_df
    
    def create_visualizations(self, results, importance_df, X_test, y_test):
        """
        Create comprehensive visualizations
        """
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # ROC Curves
        axes[0, 0].figure
        for name in results.keys():
            fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
            auc = results[name]['auc_score']
            axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature Importance Comparison (Top 10)
        top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10).index
        importance_pivot = importance_df[importance_df['feature'].isin(top_features)].pivot(
            index='feature', columns='model', values='importance'
        ).fillna(0)
        
        importance_pivot.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Top 10 Feature Importance Comparison')
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('Importance')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion Matrix - Random Forest
        cm_rf = confusion_matrix(y_test, results['Random Forest']['predictions'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
        axes[0, 2].set_title('Random Forest - Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # Confusion Matrix - XGBoost
        cm_xgb = confusion_matrix(y_test, results['XGBoost']['predictions'])
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
        axes[1, 0].set_title('XGBoost - Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Probability Distribution
        for i, (name, color) in enumerate([('Random Forest', 'blue'), ('XGBoost', 'green')]):
            probas = results[name]['probabilities']
            axes[1, 1].hist(probas[y_test == 0], alpha=0.5, label=f'{name} - No Default', 
                           color=color, bins=30)
            axes[1, 1].hist(probas[y_test == 1], alpha=0.5, label=f'{name} - Default', 
                           color=color, bins=30, histtype='step', linewidth=2)
        
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distributions')
        axes[1, 1].legend()
        
        # Model Performance Comparison
        models = list(results.keys())
        auc_scores = [results[model]['auc_score'] for model in models]
        
        bars = axes[1, 2].bar(models, auc_scores, color=['skyblue', 'lightgreen'])
        axes[1, 2].set_title('Model Performance Comparison (AUC)')
        axes[1, 2].set_ylabel('AUC Score')
        axes[1, 2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ensemble_learning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual feature importance plots
        plt.figure(figsize=(12, 8))
        
        # Random Forest Feature Importance
        plt.subplot(1, 2, 1)
        rf_top = importance_df[importance_df['model'] == 'Random Forest'].head(10)
        plt.barh(rf_top['feature'], rf_top['importance'])
        plt.title('Random Forest - Top 10 Features')
        plt.xlabel('Feature Importance')
        
        # XGBoost Feature Importance
        plt.subplot(1, 2, 2)
        xgb_top = importance_df[importance_df['model'] == 'XGBoost'].head(10)
        plt.barh(xgb_top['feature'], xgb_top['importance'])
        plt.title('XGBoost - Top 10 Features')
        plt.xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_insights(self, results, importance_df):
        """
        Save detailed model insights and comparison
        """
        insights = {
            'timestamp': datetime.now().isoformat(),
            'model_comparison': {},
            'key_insights': []
        }
        
        # Model performance comparison
        for name in results.keys():
            insights['model_comparison'][name] = {
                'auc_score': float(results[name]['auc_score']),
                'classification_report': results[name]['classification_report']
            }
        
        # Key insights
        best_model = max(results.keys(), key=lambda x: results[x]['auc_score'])
        insights['key_insights'].extend([
            f"Best performing model: {best_model}",
            f"AUC difference: {abs(results['Random Forest']['auc_score'] - results['XGBoost']['auc_score']):.4f}",
            "Top 5 most important features across both models:",
        ])
        
        # Top features
        top_features = importance_df.groupby('feature')['importance'].mean().nlargest(5)
        for feature, importance in top_features.items():
            insights['key_insights'].append(f"  - {feature}: {importance:.4f}")
        
        # Save insights
        import json
        with open(f'{self.output_dir}/model_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\nModel insights saved to {self.output_dir}/model_insights.json")
    
    def run_complete_analysis(self):
        """
        Run the complete ensemble learning analysis
        """
        print("🚀 Starting Loan Default Prediction with Ensemble Learning")
        print("=" * 60)
        
        # Generate dataset
        df = self.generate_synthetic_loan_data()
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance()
        
        # Create visualizations
        self.create_visualizations(results, importance_df, X_test, y_test)
        
        # Save insights
        self.save_model_insights(results, importance_df)
        
        print("\n✅ Analysis complete! Check the outputs folder for results.")
        print("\n📊 Key Takeaways:")
        print("- Random Forest uses bagging (parallel training of trees)")
        print("- XGBoost uses boosting (sequential learning from mistakes)")
        print("- Both provide feature importance but through different mechanisms")
        print("- Ensemble methods typically outperform single models")

if __name__ == "__main__":
    # Initialize and run analysis
    predictor = LoanDefaultPredictor()
    predictor.run_complete_analysis()