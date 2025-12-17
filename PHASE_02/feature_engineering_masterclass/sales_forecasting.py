"""
Sales Forecasting with Advanced Feature Engineering
Demonstrates the power of feature engineering: encoding, scaling, and selection techniques

"Models don't win competitions—features do."

Author: Your Name
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                 OneHotEncoder, LabelEncoder, PolynomialFeatures)
from sklearn.feature_selection import (SelectKBest, f_regression, RFE, 
                                     SelectFromModel, VarianceThreshold)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import category_encoders as ce

warnings.filterwarnings('ignore')

class SalesForecaster:
    """
    Advanced Feature Engineering pipeline for sales forecasting
    Demonstrates encoding, scaling, and feature selection techniques
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_engineering_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_sales_dataset(self, n_samples=5000):
        """
        Generate realistic sales dataset with various feature types
        """
        print("Generating comprehensive sales dataset...")
        
        np.random.seed(42)
        
        # Date range
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(n_samples)]
        
        data = {
            'date': dates,
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'quarter': [((d.month - 1) // 3) + 1 for d in dates],
            'year': [d.year for d in dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
            'is_holiday': np.random.binomial(1, 0.05, n_samples),  # 5% holiday rate
        }
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books', 'Toys']
        data['product_category'] = np.random.choice(categories, n_samples, 
                                                  p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])
        
        # Store information
        store_types = ['Mall', 'Standalone', 'Online', 'Outlet']
        data['store_type'] = np.random.choice(store_types, n_samples,
                                            p=[0.3, 0.3, 0.25, 0.15])
        
        data['store_size'] = np.random.choice(['Small', 'Medium', 'Large'], n_samples,
                                            p=[0.3, 0.5, 0.2])
        
        # Geographic data
        regions = ['North', 'South', 'East', 'West', 'Central']
        data['region'] = np.random.choice(regions, n_samples)
        
        cities = [f'City_{i}' for i in range(1, 21)]  # 20 cities
        data['city'] = np.random.choice(cities, n_samples)
        
        # Customer demographics
        data['avg_customer_age'] = np.random.normal(35, 12, n_samples)
        data['avg_customer_income'] = np.random.lognormal(10.5, 0.5, n_samples)
        
        # Marketing and promotions
        data['promotion_active'] = np.random.binomial(1, 0.3, n_samples)
        data['promotion_discount'] = np.where(data['promotion_active'], 
                                            np.random.uniform(0.05, 0.5, n_samples), 0)
        
        data['marketing_spend'] = np.random.exponential(1000, n_samples)
        data['social_media_engagement'] = np.random.gamma(2, 100, n_samples)
        
        # Weather and external factors
        data['temperature'] = np.random.normal(20, 15, n_samples)  # Celsius
        data['precipitation'] = np.random.exponential(2, n_samples)  # mm
        data['consumer_confidence_index'] = np.random.normal(100, 10, n_samples)
        
        # Competitor data
        data['competitor_count'] = np.random.poisson(3, n_samples)
        data['competitor_avg_price'] = np.random.lognormal(3, 0.3, n_samples)
        
        # Product features
        data['product_price'] = np.random.lognormal(3.5, 0.8, n_samples)
        data['product_rating'] = np.random.beta(8, 2, n_samples) * 5  # 0-5 rating
        data['product_reviews_count'] = np.random.poisson(50, n_samples)
        data['inventory_level'] = np.random.lognormal(4, 0.5, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Engineer target variable with realistic relationships
        sales = (
            # Base sales
            100 +
            
            # Seasonal patterns
            50 * np.sin(2 * np.pi * df['month'] / 12) +  # Monthly seasonality
            30 * (df['day_of_week'] <= 4).astype(int) +   # Weekday effect
            100 * df['is_weekend'] +                      # Weekend boost
            200 * df['is_holiday'] +                      # Holiday effect
            
            # Product and store effects
            df['product_category'].map({
                'Electronics': 150, 'Clothing': 100, 'Home': 120,
                'Sports': 90, 'Books': 60, 'Toys': 80
            }) +
            df['store_type'].map({
                'Mall': 100, 'Standalone': 80, 'Online': 120, 'Outlet': 60
            }) +
            df['store_size'].map({'Small': 50, 'Medium': 100, 'Large': 150}) +
            
            # Marketing and promotions
            300 * df['promotion_active'] +
            500 * df['promotion_discount'] +
            0.1 * df['marketing_spend'] +
            0.05 * df['social_media_engagement'] +
            
            # Economic and external factors
            2 * df['temperature'] +
            -10 * df['precipitation'] +
            1.5 * df['consumer_confidence_index'] +
            
            # Product factors
            -50 * np.log(df['product_price']) +
            100 * df['product_rating'] +
            0.2 * df['product_reviews_count'] +
            0.01 * df['inventory_level'] +
            
            # Regional effects
            df['region'].map({
                'North': 50, 'South': 30, 'East': 40, 'West': 60, 'Central': 35
            }) +
            
            # Random noise
            np.random.normal(0, 50, n_samples)
        )
        
        # Ensure positive sales
        df['sales'] = np.maximum(sales, 10)
        
        print(f"Dataset created with {len(df)} samples")
        print(f"Sales statistics: Mean={df['sales'].mean():.2f}, Std={df['sales'].std():.2f}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def basic_feature_engineering(self, df):
        """
        Create basic engineered features
        """
        print("\n=== Basic Feature Engineering ===")
        
        df = df.copy()
        
        # Date-based features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Cyclical encoding for periodic features
        for period, col in [(7, 'day_of_week'), (12, 'month'), (365, 'day_of_year')]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
        
        # Interaction features
        df['price_rating_interaction'] = df['product_price'] * df['product_rating']
        df['marketing_per_competitor'] = df['marketing_spend'] / (df['competitor_count'] + 1)
        df['promotion_price_ratio'] = df['promotion_discount'] / df['product_price']
        
        # Aggregation features (rolling statistics)
        df = df.sort_values('date').reset_index(drop=True)
        
        # 7-day rolling features
        for col in ['sales', 'marketing_spend', 'temperature']:
            if col != 'sales':  # Don't use target for feature engineering
                df[f'{col}_7day_mean'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_7day_std'] = df[col].rolling(window=7, min_periods=1).std()
        
        # Lag features (previous values)
        for lag in [1, 7, 30]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            df[f'marketing_lag_{lag}'] = df['marketing_spend'].shift(lag)
        
        # Fill NaN values from lag features
        df.fillna(method='bfill', inplace=True)
        df.fillna(df.mean(), inplace=True)
        
        print(f"Features after basic engineering: {len(df.columns)}")
        
        return df
    
    def advanced_categorical_encoding(self, df, target_col='sales'):
        """
        Demonstrate various categorical encoding techniques
        """
        print("\n=== Advanced Categorical Encoding ===")
        
        df = df.copy()
        encoding_results = {}
        
        categorical_cols = ['product_category', 'store_type', 'store_size', 'region', 'city']
        
        # 1. One-Hot Encoding
        print("Applying One-Hot Encoding...")
        df_encoded = pd.get_dummies(df, columns=[col + '_onehot' for col in categorical_cols], 
                                  prefix=[col for col in categorical_cols])
        
        # For demonstration, create separate columns for one-hot
        for col in categorical_cols:
            df[col + '_onehot'] = df[col]
        
        onehot_df = pd.get_dummies(df, columns=[col + '_onehot' for col in categorical_cols])
        encoding_results['onehot_features'] = len(onehot_df.columns) - len(df.columns)
        
        # 2. Label Encoding
        print("Applying Label Encoding...")
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_label'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        self.encoders['label'] = label_encoders
        
        # 3. Target Encoding (Mean Encoding)
        print("Applying Target Encoding...")
        target_encoders = {}
        for col in categorical_cols:
            target_mean = df.groupby(col)[target_col].mean()
            df[col + '_target'] = df[col].map(target_mean)
            target_encoders[col] = target_mean
        
        self.encoders['target'] = target_encoders
        
        # 4. Frequency Encoding
        print("Applying Frequency Encoding...")
        for col in categorical_cols:
            freq_map = df[col].value_counts(normalize=True)
            df[col + '_freq'] = df[col].map(freq_map)
        
        # 5. Binary Encoding
        print("Applying Binary Encoding...")
        binary_encoder = ce.BinaryEncoder(cols=categorical_cols, return_df=True)
        binary_encoded = binary_encoder.fit_transform(df[categorical_cols])
        
        # Add binary encoded columns
        for col in binary_encoded.columns:
            df[f'binary_{col}'] = binary_encoded[col]
        
        self.encoders['binary'] = binary_encoder
        
        encoding_results.update({
            'original_features': len([c for c in df.columns if c in categorical_cols]),
            'label_encoded': len(categorical_cols),
            'target_encoded': len(categorical_cols),
            'frequency_encoded': len(categorical_cols),
            'binary_encoded': len(binary_encoded.columns)
        })
        
        print(f"Encoding results: {encoding_results}")
        
        return df, encoding_results
    
    def feature_scaling_comparison(self, X_train, X_test, numerical_cols):
        """
        Compare different scaling techniques
        """
        print("\n=== Feature Scaling Comparison ===")
        
        scaling_methods = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'No_Scaling': None
        }
        
        scaled_datasets = {}
        scaling_stats = {}
        
        for name, scaler in scaling_methods.items():
            print(f"Applying {name}...")
            
            if scaler is not None:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                
                X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
                X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
                
                self.scalers[name] = scaler
                
                # Calculate scaling statistics
                scaling_stats[name] = {
                    'mean': float(X_train_scaled[numerical_cols].mean().mean()),
                    'std': float(X_train_scaled[numerical_cols].std().mean()),
                    'min': float(X_train_scaled[numerical_cols].min().min()),
                    'max': float(X_train_scaled[numerical_cols].max().max())
                }
                
            else:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                
                scaling_stats[name] = {
                    'mean': float(X_train[numerical_cols].mean().mean()),
                    'std': float(X_train[numerical_cols].std().mean()),
                    'min': float(X_train[numerical_cols].min().min()),
                    'max': float(X_train[numerical_cols].max().max())
                }
            
            scaled_datasets[name] = (X_train_scaled, X_test_scaled)
        
        print(f"Scaling comparison complete for {len(numerical_cols)} numerical features")
        
        return scaled_datasets, scaling_stats
    
    def advanced_feature_selection(self, X_train, y_train, X_test):
        """
        Demonstrate various feature selection techniques
        """
        print("\n=== Advanced Feature Selection ===")
        
        selection_methods = {}
        selected_features = {}
        
        # 1. Variance Threshold
        print("Applying Variance Threshold...")
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X_train)
        selected_features['variance'] = X_train.columns[variance_selector.get_support()].tolist()
        selection_methods['variance'] = variance_selector
        
        # 2. Univariate Selection (K-Best)
        print("Applying K-Best Selection...")
        k_best_selector = SelectKBest(score_func=f_regression, k=min(20, X_train.shape[1]))
        X_kbest = k_best_selector.fit_transform(X_train, y_train)
        selected_features['k_best'] = X_train.columns[k_best_selector.get_support()].tolist()
        selection_methods['k_best'] = k_best_selector
        
        # 3. Recursive Feature Elimination (RFE)
        print("Applying Recursive Feature Elimination...")
        estimator = RandomForestRegressor(n_estimators=20, random_state=42)
        rfe_selector = RFE(estimator=estimator, n_features_to_select=min(15, X_train.shape[1]))
        X_rfe = rfe_selector.fit_transform(X_train, y_train)
        selected_features['rfe'] = X_train.columns[rfe_selector.get_support()].tolist()
        selection_methods['rfe'] = rfe_selector
        
        # 4. L1-based Selection (Lasso)
        print("Applying L1-based Selection...")
        lasso_selector = SelectFromModel(Lasso(alpha=0.1, random_state=42))
        X_lasso = lasso_selector.fit_transform(X_train, y_train)
        selected_features['lasso'] = X_train.columns[lasso_selector.get_support()].tolist()
        selection_methods['lasso'] = lasso_selector
        
        # 5. Tree-based Selection (Random Forest)
        print("Applying Tree-based Selection...")
        rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))
        X_rf = rf_selector.fit_transform(X_train, y_train)
        selected_features['random_forest'] = X_train.columns[rf_selector.get_support()].tolist()
        selection_methods['random_forest'] = rf_selector
        
        self.feature_selectors = selection_methods
        
        # Summary statistics
        selection_summary = {
            method: len(features) for method, features in selected_features.items()
        }
        
        print(f"Feature selection summary: {selection_summary}")
        print(f"Original features: {X_train.shape[1]}")
        
        return selected_features, selection_summary
    
    def train_models_with_different_features(self, datasets, y_train, y_test, feature_sets):
        """
        Train models with different feature engineering approaches
        """
        print("\n=== Training Models with Different Feature Sets ===")
        
        model_configs = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        # Test different scaling methods
        for scale_name, (X_train_scaled, X_test_scaled) in datasets.items():
            scale_results = {}
            
            # Test different feature selection methods
            for selection_name, features in feature_sets.items():
                if len(features) == 0:
                    continue
                
                # Get features that exist in the scaled dataset
                available_features = [f for f in features if f in X_train_scaled.columns]
                if len(available_features) == 0:
                    continue
                
                X_train_selected = X_train_scaled[available_features]
                X_test_selected = X_test_scaled[available_features]
                
                selection_results = {}
                
                for model_name, model in model_configs.items():
                    try:
                        # Train model
                        model.fit(X_train_selected, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test_selected)
                        
                        # Metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        selection_results[model_name] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'n_features': len(available_features)
                        }
                        
                    except Exception as e:
                        print(f"Error training {model_name} with {scale_name}/{selection_name}: {e}")
                        continue
                
                if selection_results:
                    scale_results[selection_name] = selection_results
            
            if scale_results:
                results[scale_name] = scale_results
        
        # Also test with all features (no selection)
        for scale_name, (X_train_scaled, X_test_scaled) in datasets.items():
            if 'all_features' not in results.get(scale_name, {}):
                if scale_name not in results:
                    results[scale_name] = {}
                
                all_features_results = {}
                
                for model_name, model in model_configs.items():
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        all_features_results[model_name] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'n_features': X_train_scaled.shape[1]
                        }
                    except Exception as e:
                        print(f"Error training {model_name} with all features: {e}")
                        continue
                
                results[scale_name]['all_features'] = all_features_results
        
        print(f"Model training complete for {len(results)} scaling methods")
        
        return results
    
    def create_comprehensive_visualizations(self, results, scaling_stats, encoding_results):
        """
        Create comprehensive visualizations for feature engineering analysis
        """
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Scaling Methods Comparison
        axes[0, 0].bar(scaling_stats.keys(), [s['std'] for s in scaling_stats.values()])
        axes[0, 0].set_title('Standard Deviation by Scaling Method')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Encoding Methods Feature Count
        encoding_counts = {
            'Original': encoding_results['original_features'],
            'Label': encoding_results['label_encoded'],
            'Target': encoding_results['target_encoded'],
            'Frequency': encoding_results['frequency_encoded'],
            'Binary': encoding_results['binary_encoded'],
            'One-Hot': encoding_results.get('onehot_features', 0)
        }
        
        axes[0, 1].bar(encoding_counts.keys(), encoding_counts.values(), color='lightblue')
        axes[0, 1].set_title('Features Created by Encoding Method')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Model Performance Heatmap (R² scores)
        performance_data = []
        scale_methods = []
        selection_methods = []
        
        for scale_name, scale_results in results.items():
            for selection_name, selection_results in scale_results.items():
                for model_name, metrics in selection_results.items():
                    performance_data.append({
                        'Scale': scale_name,
                        'Selection': selection_name,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'RMSE': metrics['rmse'],
                        'Features': metrics['n_features']
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Create pivot for heatmap
            heatmap_data = perf_df.pivot_table(index=['Scale', 'Selection'], 
                                             columns='Model', values='R²')
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                       ax=axes[0, 2], cbar_kws={'label': 'R² Score'})
            axes[0, 2].set_title('Model Performance by Feature Engineering')
        
        # 4. Feature Count vs Performance
        if performance_data:
            axes[1, 0].scatter(perf_df['Features'], perf_df['R²'], alpha=0.6, c=perf_df['RMSE'], 
                              cmap='viridis_r')
            axes[1, 0].set_xlabel('Number of Features')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].set_title('Feature Count vs Model Performance')
            
            # Best performance annotation
            best_idx = perf_df['R²'].idxmax()
            best_row = perf_df.loc[best_idx]
            axes[1, 0].annotate(f"Best: {best_row['Model']}\n{best_row['Scale']}/{best_row['Selection']}",
                               xy=(best_row['Features'], best_row['R²']),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 5. Scaling Method Impact
        scale_performance = perf_df.groupby(['Scale', 'Model'])['R²'].mean().unstack()
        scale_performance.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average R² by Scaling Method')
        axes[1, 1].set_ylabel('Average R² Score')
        axes[1, 1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Feature Selection Impact
        selection_performance = perf_df.groupby(['Selection', 'Model'])['R²'].mean().unstack()
        selection_performance.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Average R² by Feature Selection')
        axes[1, 2].set_ylabel('Average R² Score')
        axes[1, 2].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. RMSE vs R² Trade-off
        axes[2, 0].scatter(perf_df['RMSE'], perf_df['R²'], alpha=0.6, c=perf_df['Features'],
                          cmap='plasma')
        axes[2, 0].set_xlabel('RMSE')
        axes[2, 0].set_ylabel('R² Score')
        axes[2, 0].set_title('RMSE vs R² Trade-off')
        colorbar = plt.colorbar(axes[2, 0].collections[0], ax=axes[2, 0])
        colorbar.set_label('Number of Features')
        
        # 8. Model Comparison
        model_avg = perf_df.groupby('Model')[['R²', 'RMSE']].mean()
        x_pos = np.arange(len(model_avg))
        
        ax2_twin = axes[2, 1].twinx()
        bars1 = axes[2, 1].bar(x_pos - 0.2, model_avg['R²'], 0.4, label='R²', color='skyblue')
        bars2 = ax2_twin.bar(x_pos + 0.2, model_avg['RMSE'], 0.4, label='RMSE', color='salmon')
        
        axes[2, 1].set_xlabel('Models')
        axes[2, 1].set_ylabel('R² Score')
        ax2_twin.set_ylabel('RMSE')
        axes[2, 1].set_title('Average Model Performance')
        axes[2, 1].set_xticks(x_pos)
        axes[2, 1].set_xticklabels(model_avg.index, rotation=45)
        
        # Combined legend
        lines1, labels1 = axes[2, 1].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[2, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 9. Feature Engineering Impact Summary
        summary_data = {
            'No Engineering': perf_df[perf_df['Selection'] == 'all_features']['R²'].mean(),
            'With Selection': perf_df[perf_df['Selection'] != 'all_features']['R²'].mean(),
            'Best Combination': perf_df['R²'].max()
        }
        
        bars = axes[2, 2].bar(summary_data.keys(), summary_data.values(), 
                             color=['lightcoral', 'lightblue', 'lightgreen'])
        axes[2, 2].set_title('Feature Engineering Impact')
        axes[2, 2].set_ylabel('R² Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, summary_data.values()):
            axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_engineering_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        if performance_data:
            detailed_df = pd.DataFrame(performance_data)
            detailed_df.to_csv(f'{self.output_dir}/feature_engineering_results.csv', index=False)
    
    def generate_insights_report(self, results, encoding_results):
        """
        Generate comprehensive insights about feature engineering
        """
        print("\n=== Generating Insights Report ===")
        
        performance_data = []
        for scale_name, scale_results in results.items():
            for selection_name, selection_results in scale_results.items():
                for model_name, metrics in selection_results.items():
                    performance_data.append({
                        'scale': scale_name,
                        'selection': selection_name,
                        'model': model_name,
                        'r2': metrics['r2'],
                        'rmse': metrics['rmse'],
                        'features': metrics['n_features']
                    })
        
        if not performance_data:
            print("No performance data available for insights.")
            return
        
        perf_df = pd.DataFrame(performance_data)
        
        # Find best combinations
        best_overall = perf_df.loc[perf_df['r2'].idxmax()]
        best_per_model = perf_df.groupby('model')['r2'].idxmax()
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_features_after_engineering': len(encoding_results),
                'encoding_methods_used': list(encoding_results.keys())
            },
            'best_performance': {
                'overall': {
                    'scaling': best_overall['scale'],
                    'selection': best_overall['selection'],
                    'model': best_overall['model'],
                    'r2_score': float(best_overall['r2']),
                    'rmse': float(best_overall['rmse']),
                    'n_features': int(best_overall['features'])
                }
            },
            'model_rankings': {},
            'feature_engineering_impact': {},
            'recommendations': []
        }
        
        # Model rankings
        model_avg_r2 = perf_df.groupby('model')['r2'].mean().sort_values(ascending=False)
        insights['model_rankings'] = {
            model: float(score) for model, score in model_avg_r2.items()
        }
        
        # Feature engineering impact analysis
        no_selection = perf_df[perf_df['selection'] == 'all_features']['r2'].mean()
        with_selection = perf_df[perf_df['selection'] != 'all_features']['r2'].mean()
        
        insights['feature_engineering_impact'] = {
            'baseline_r2': float(no_selection) if not pd.isna(no_selection) else 0.0,
            'optimized_r2': float(with_selection) if not pd.isna(with_selection) else 0.0,
            'improvement': float(with_selection - no_selection) if not pd.isna(with_selection) and not pd.isna(no_selection) else 0.0
        }
        
        # Generate recommendations
        insights['recommendations'].extend([
            f"Best model: {best_overall['model']} with {best_overall['scale']} scaling",
            f"Optimal feature count: {best_overall['features']} features",
            f"Feature selection improved performance by {insights['feature_engineering_impact']['improvement']:.3f}",
            f"Top performing scaling method: {perf_df.groupby('scale')['r2'].mean().idxmax()}",
            f"Most consistent model: {perf_df.groupby('model')['r2'].std().idxmin()}"
        ])
        
        # Save insights
        import json
        with open(f'{self.output_dir}/feature_engineering_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"✅ Insights saved to {self.output_dir}/feature_engineering_insights.json")
        
        return insights
    
    def run_complete_analysis(self):
        """
        Run the complete feature engineering analysis
        """
        print("🚀 Starting Sales Forecasting with Feature Engineering Masterclass")
        print("=" * 70)
        
        # Generate dataset
        df = self.generate_sales_dataset()
        
        # Basic feature engineering
        df_engineered = self.basic_feature_engineering(df)
        
        # Advanced categorical encoding
        df_encoded, encoding_results = self.advanced_categorical_encoding(df_engineered)
        
        # Prepare data for modeling
        target_col = 'sales'
        feature_cols = [col for col in df_encoded.columns 
                       if col not in ['date', target_col] and not col.startswith('sales_lag')]
        
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        # Time-based split for time series data
        split_date = df_encoded['date'].quantile(0.8)
        train_mask = df_encoded['date'] <= split_date
        
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Total features: {len(feature_cols)}")
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Feature scaling comparison
        scaled_datasets, scaling_stats = self.feature_scaling_comparison(
            X_train, X_test, numerical_cols
        )
        
        # Feature selection
        selected_features, selection_summary = self.advanced_feature_selection(
            X_train, y_train, X_test
        )
        
        # Train models with different configurations
        results = self.train_models_with_different_features(
            scaled_datasets, y_train, y_test, selected_features
        )
        
        # Create visualizations
        self.create_comprehensive_visualizations(results, scaling_stats, encoding_results)
        
        # Generate insights
        insights = self.generate_insights_report(results, encoding_results)
        
        print("\n✅ Feature Engineering Masterclass Complete!")
        print("\n📊 Key Takeaways:")
        print("- Feature engineering can dramatically improve model performance")
        print("- Different models respond differently to scaling and selection")
        print("- The right combination of techniques matters more than individual methods")
        print("- More features ≠ better performance (curse of dimensionality)")
        
        return insights

if __name__ == "__main__":
    # Initialize and run analysis
    forecaster = SalesForecaster()
    forecaster.run_complete_analysis()