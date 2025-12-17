"""
Customer Segmentation Using K-Means & DBSCAN
Demonstrates unsupervised learning: clustering algorithms, elbow method, and silhouette analysis

"No labels? No problem."

Author: Your Name
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import warnings
import os
from datetime import datetime, timedelta
import json
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Comprehensive customer segmentation analysis using multiple clustering algorithms
    Demonstrates K-Means, DBSCAN, and hierarchical clustering with evaluation metrics
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.clustering_results = {}
        self.optimal_params = {}
        self.customer_profiles = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_customer_dataset(self, n_customers=3000):
        """
        Generate realistic customer dataset with multiple behavioral patterns
        """
        print("Generating customer dataset...")
        
        np.random.seed(42)
        
        # Generate customer segments with different behaviors
        segments = []
        
        # Segment 1: High-value customers (20%)
        n_segment1 = int(0.20 * n_customers)
        segment1 = {
            'customer_id': range(1, n_segment1 + 1),
            'annual_spending': np.random.normal(5000, 1200, n_segment1),
            'frequency': np.random.poisson(50, n_segment1),
            'recency': np.random.exponential(15, n_segment1),
            'avg_order_value': np.random.normal(150, 30, n_segment1),
            'customer_lifetime_months': np.random.normal(36, 12, n_segment1),
            'return_rate': np.random.beta(2, 3, n_segment1) * 0.3,
            'support_tickets': np.random.poisson(2, n_segment1),
            'product_categories': np.random.poisson(4, n_segment1),
            'digital_engagement': np.random.beta(8, 2, n_segment1) * 100,
            'segment_true': 'High Value'
        }
        segments.append(pd.DataFrame(segment1))
        
        # Segment 2: Regular customers (35%)
        n_segment2 = int(0.35 * n_customers)
        segment2 = {
            'customer_id': range(n_segment1 + 1, n_segment1 + n_segment2 + 1),
            'annual_spending': np.random.normal(2500, 600, n_segment2),
            'frequency': np.random.poisson(25, n_segment2),
            'recency': np.random.exponential(30, n_segment2),
            'avg_order_value': np.random.normal(100, 25, n_segment2),
            'customer_lifetime_months': np.random.normal(24, 8, n_segment2),
            'return_rate': np.random.beta(2, 5, n_segment2) * 0.3,
            'support_tickets': np.random.poisson(1, n_segment2),
            'product_categories': np.random.poisson(2.5, n_segment2),
            'digital_engagement': np.random.beta(5, 3, n_segment2) * 100,
            'segment_true': 'Regular'
        }
        segments.append(pd.DataFrame(segment2))
        
        # Segment 3: Occasional customers (30%)
        n_segment3 = int(0.30 * n_customers)
        segment3 = {
            'customer_id': range(n_segment1 + n_segment2 + 1, n_segment1 + n_segment2 + n_segment3 + 1),
            'annual_spending': np.random.normal(800, 300, n_segment3),
            'frequency': np.random.poisson(8, n_segment3),
            'recency': np.random.exponential(60, n_segment3),
            'avg_order_value': np.random.normal(75, 20, n_segment3),
            'customer_lifetime_months': np.random.normal(12, 6, n_segment3),
            'return_rate': np.random.beta(1, 8, n_segment3) * 0.3,
            'support_tickets': np.random.poisson(0.5, n_segment3),
            'product_categories': np.random.poisson(1.5, n_segment3),
            'digital_engagement': np.random.beta(2, 6, n_segment3) * 100,
            'segment_true': 'Occasional'
        }
        segments.append(pd.DataFrame(segment3))
        
        # Segment 4: At-risk customers (15%)
        n_segment4 = n_customers - n_segment1 - n_segment2 - n_segment3
        segment4 = {
            'customer_id': range(n_segment1 + n_segment2 + n_segment3 + 1, n_customers + 1),
            'annual_spending': np.random.normal(1200, 400, n_segment4),
            'frequency': np.random.poisson(5, n_segment4),
            'recency': np.random.exponential(120, n_segment4),
            'avg_order_value': np.random.normal(60, 15, n_segment4),
            'customer_lifetime_months': np.random.normal(18, 10, n_segment4),
            'return_rate': np.random.beta(1, 4, n_segment4) * 0.3,
            'support_tickets': np.random.poisson(3, n_segment4),
            'product_categories': np.random.poisson(1, n_segment4),
            'digital_engagement': np.random.beta(1, 9, n_segment4) * 100,
            'segment_true': 'At Risk'
        }
        segments.append(pd.DataFrame(segment4))
        
        # Combine all segments
        df = pd.concat(segments, ignore_index=True)
        
        # Ensure positive values
        numerical_cols = ['annual_spending', 'frequency', 'recency', 'avg_order_value', 
                         'customer_lifetime_months', 'product_categories']
        for col in numerical_cols:
            df[col] = np.maximum(df[col], 1)
        
        # Add some derived features
        df['monetary_recency_ratio'] = df['annual_spending'] / (df['recency'] + 1)
        df['engagement_score'] = (df['frequency'] * df['digital_engagement']) / 100
        df['loyalty_score'] = df['customer_lifetime_months'] * (1 - df['return_rate'])
        df['ticket_per_month'] = df['support_tickets'] / (df['customer_lifetime_months'] + 1)
        
        print(f"Dataset created with {len(df)} customers")
        print(f"True segment distribution:")
        print(df['segment_true'].value_counts())
        
        return df
    
    def analyze_customer_features(self, df):
        """
        Analyze customer features and their distributions
        """
        print("\n=== Customer Feature Analysis ===")
        
        # Select features for clustering (exclude IDs and true labels)
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id', 'segment_true']]
        
        # Basic statistics
        print("Feature Statistics:")
        print(df[feature_cols].describe())
        
        # Correlation analysis
        correlation_matrix = df[feature_cols].corr()
        
        # Create feature distribution plots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_cols[:12]):
            axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title('Customer Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_cols, correlation_matrix
    
    def elbow_method_analysis(self, X_scaled, max_k=10):
        """
        Perform elbow method analysis to find optimal number of clusters
        """
        print("\n=== Elbow Method Analysis ===")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate inertia (within-cluster sum of squares)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(sil_score)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for potential elbow points
        for i, (k, inertia) in enumerate(zip(k_range, inertias)):
            ax1.annotate(f'k={k}', (k, inertia), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True, alpha=0.3)
        
        # Mark best silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        ax2.annotate(f'Best: k={best_k}\nScore={best_score:.3f}', 
                    xy=(best_k, best_score), xytext=(best_k+0.5, best_score-0.02),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/elbow_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find elbow point using rate of change
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        elbow_k = k_range[np.argmax(second_deltas) + 2]  # +2 due to double diff
        
        self.optimal_params['elbow_k'] = elbow_k
        self.optimal_params['silhouette_best_k'] = best_k
        
        print(f"Suggested k from elbow method: {elbow_k}")
        print(f"Best k from silhouette analysis: {best_k}")
        
        return elbow_k, best_k, inertias, silhouette_scores
    
    def dbscan_parameter_tuning(self, X_scaled):
        """
        Find optimal DBSCAN parameters using k-distance plot
        """
        print("\n=== DBSCAN Parameter Tuning ===")
        
        # Calculate k-distance plot for epsilon selection
        k = 4  # MinPts - 1 (rule of thumb: 2 * dimensions)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)
        
        # Sort distances to k-th nearest neighbor
        distances = np.sort(distances, axis=0)
        distances = distances[:, k-1]
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel('Data Points Sorted by Distance')
        plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
        plt.title('K-Distance Plot for DBSCAN Epsilon Selection')
        plt.grid(True, alpha=0.3)
        
        # Find knee point (elbow) in k-distance plot
        # Use simple derivative approach
        diff = np.diff(distances)
        knee_index = np.argmax(diff)
        optimal_eps = distances[knee_index]
        
        plt.axhline(y=optimal_eps, color='red', linestyle='--', 
                   label=f'Suggested ε = {optimal_eps:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dbscan_kdistance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Test different epsilon values around the suggested one
        eps_range = np.linspace(optimal_eps * 0.5, optimal_eps * 1.5, 10)
        min_samples_range = [3, 4, 5, 6, 8, 10]
        
        best_eps = optimal_eps
        best_min_samples = 4
        best_score = -1
        
        results = []
        
        print("Testing different DBSCAN parameters...")
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                
                # Skip if no clusters found or all points are noise
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(labels) * 0.5:  # Less than 50% noise
                    sil_score = silhouette_score(X_scaled, labels)
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': sil_score
                    })
                    
                    if sil_score > best_score:
                        best_score = sil_score
                        best_eps = eps
                        best_min_samples = min_samples
        
        self.optimal_params['dbscan_eps'] = best_eps
        self.optimal_params['dbscan_min_samples'] = best_min_samples
        
        print(f"Optimal DBSCAN parameters:")
        print(f"  ε (epsilon): {best_eps:.3f}")
        print(f"  min_samples: {best_min_samples}")
        print(f"  Best silhouette score: {best_score:.3f}")
        
        return best_eps, best_min_samples, results
    
    def perform_clustering(self, X_scaled, optimal_k):
        """
        Perform clustering using different algorithms
        """
        print("\n=== Performing Clustering ===")
        
        clustering_algorithms = {}
        
        # 1. K-Means Clustering
        print("Running K-Means clustering...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        clustering_algorithms['K-Means'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_
        }
        
        # 2. DBSCAN Clustering
        print("Running DBSCAN clustering...")
        dbscan = DBSCAN(eps=self.optimal_params['dbscan_eps'], 
                       min_samples=self.optimal_params['dbscan_min_samples'])
        dbscan_labels = dbscan.fit_predict(X_scaled)
        clustering_algorithms['DBSCAN'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'core_samples': dbscan.core_sample_indices_
        }
        
        # 3. Hierarchical Clustering (Agglomerative)
        print("Running Hierarchical clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        clustering_algorithms['Hierarchical'] = {
            'model': hierarchical,
            'labels': hierarchical_labels
        }
        
        # Calculate clustering metrics
        for name, result in clustering_algorithms.items():
            labels = result['labels']
            
            # Skip silhouette for DBSCAN if too few clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                sil_score = silhouette_score(X_scaled, labels)
                result['silhouette_score'] = sil_score
                result['n_clusters'] = n_clusters
                
                if name == 'DBSCAN':
                    result['n_noise'] = list(labels).count(-1)
                    result['noise_ratio'] = result['n_noise'] / len(labels)
                
                print(f"{name}: {n_clusters} clusters, Silhouette = {sil_score:.3f}")
                if name == 'DBSCAN':
                    print(f"  Noise points: {result['n_noise']} ({result['noise_ratio']:.1%})")
        
        self.clustering_results = clustering_algorithms
        return clustering_algorithms
    
    def create_comprehensive_visualizations(self, df, X_scaled, feature_cols):
        """
        Create comprehensive clustering visualizations
        """
        print("\n=== Creating Comprehensive Visualizations ===")
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance[0]:.1%} + {explained_variance[1]:.1%} = {sum(explained_variance):.1%}")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Plot clustering results
        algorithms = ['K-Means', 'DBSCAN', 'Hierarchical']
        
        for i, alg_name in enumerate(algorithms):
            if alg_name in self.clustering_results:
                labels = self.clustering_results[alg_name]['labels']
                
                # Main clustering plot
                scatter = axes[0, i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                           cmap='tab10', alpha=0.7, s=50)
                axes[0, i].set_title(f'{alg_name} Clustering Results')
                axes[0, i].set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
                axes[0, i].set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
                
                # Add cluster centers for K-Means
                if alg_name == 'K-Means' and 'centers' in self.clustering_results[alg_name]:
                    centers_pca = pca.transform(self.clustering_results[alg_name]['centers'])
                    axes[0, i].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                                     marker='x', s=200, c='red', linewidths=3)
                
                plt.colorbar(scatter, ax=axes[0, i])
        
        # Silhouette analysis
        for i, alg_name in enumerate(algorithms):
            if alg_name in self.clustering_results and 'silhouette_score' in self.clustering_results[alg_name]:
                labels = self.clustering_results[alg_name]['labels']
                
                # Calculate silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(X_scaled, labels)
                
                y_lower = 10
                unique_labels = set(labels)
                if -1 in unique_labels:  # Remove noise label for DBSCAN
                    unique_labels.remove(-1)
                unique_labels = sorted(list(unique_labels))
                
                for label in unique_labels:
                    cluster_silhouette_values = sample_silhouette_values[labels == label]
                    cluster_silhouette_values.sort()
                    
                    size_cluster = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster
                    
                    color = plt.cm.tab10(label / len(unique_labels))
                    axes[1, i].fill_betweenx(np.arange(y_lower, y_upper),
                                           0, cluster_silhouette_values,
                                           facecolor=color, edgecolor=color, alpha=0.7)
                    
                    axes[1, i].text(-0.05, y_lower + 0.5 * size_cluster, str(label))
                    y_lower = y_upper + 10
                
                avg_score = self.clustering_results[alg_name]['silhouette_score']
                axes[1, i].axvline(x=avg_score, color="red", linestyle="--",
                                 label=f'Average Score: {avg_score:.3f}')
                axes[1, i].set_title(f'{alg_name} Silhouette Analysis')
                axes[1, i].set_xlabel('Silhouette Coefficient Values')
                axes[1, i].set_ylabel('Cluster Label')
                axes[1, i].legend()
        
        # Feature comparison by clusters (using K-Means)
        if 'K-Means' in self.clustering_results:
            labels = self.clustering_results['K-Means']['labels']
            
            # Select top 6 features for visualization
            top_features = feature_cols[:6]
            
            cluster_means = []
            for cluster in sorted(set(labels)):
                cluster_data = df[df.index.isin(np.where(labels == cluster)[0])]
                means = cluster_data[top_features].mean()
                cluster_means.append(means)
            
            cluster_means_df = pd.DataFrame(cluster_means, 
                                          columns=top_features,
                                          index=[f'Cluster {i}' for i in sorted(set(labels))])
            
            # Normalize for better visualization
            cluster_means_normalized = cluster_means_df.div(cluster_means_df.max(), axis=1)
            
            sns.heatmap(cluster_means_normalized.T, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0.5, ax=axes[2, 0])
            axes[2, 0].set_title('Cluster Profiles (Normalized Features)')
            axes[2, 0].set_xlabel('Clusters')
            axes[2, 0].set_ylabel('Features')
        
        # Cluster size comparison
        cluster_sizes = {}
        for alg_name in algorithms:
            if alg_name in self.clustering_results:
                labels = self.clustering_results[alg_name]['labels']
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                
                sizes = [list(labels).count(label) for label in sorted(unique_labels)]
                cluster_sizes[alg_name] = sizes
        
        # Plot cluster sizes
        algorithms_with_results = list(cluster_sizes.keys())
        x = np.arange(len(algorithms_with_results))
        width = 0.25
        
        max_clusters = max(len(sizes) for sizes in cluster_sizes.values())
        colors = plt.cm.Set3(np.linspace(0, 1, max_clusters))
        
        for i in range(max_clusters):
            cluster_counts = []
            for alg in algorithms_with_results:
                if i < len(cluster_sizes[alg]):
                    cluster_counts.append(cluster_sizes[alg][i])
                else:
                    cluster_counts.append(0)
            
            axes[2, 1].bar(x + i * width, cluster_counts, width, 
                          label=f'Cluster {i}', color=colors[i])
        
        axes[2, 1].set_xlabel('Clustering Algorithm')
        axes[2, 1].set_ylabel('Number of Customers')
        axes[2, 1].set_title('Cluster Size Comparison')
        axes[2, 1].set_xticks(x + width)
        axes[2, 1].set_xticklabels(algorithms_with_results)
        axes[2, 1].legend()
        
        # Algorithm performance comparison
        performance_data = {
            'Algorithm': [],
            'Silhouette Score': [],
            'Number of Clusters': [],
            'Noise Points': []
        }
        
        for alg_name in algorithms:
            if alg_name in self.clustering_results and 'silhouette_score' in self.clustering_results[alg_name]:
                performance_data['Algorithm'].append(alg_name)
                performance_data['Silhouette Score'].append(self.clustering_results[alg_name]['silhouette_score'])
                performance_data['Number of Clusters'].append(self.clustering_results[alg_name]['n_clusters'])
                performance_data['Noise Points'].append(self.clustering_results[alg_name].get('n_noise', 0))
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create performance table
        table_data = perf_df.values
        table = axes[2, 2].table(cellText=table_data, colLabels=perf_df.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Algorithm Performance Summary')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_clustering_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save performance results
        perf_df.to_csv(f'{self.output_dir}/clustering_performance.csv', index=False)
    
    def analyze_customer_segments(self, df, feature_cols):
        """
        Analyze and profile the discovered customer segments
        """
        print("\n=== Customer Segment Analysis ===")
        
        # Use K-Means results for detailed analysis
        if 'K-Means' not in self.clustering_results:
            print("K-Means results not available for segment analysis.")
            return
        
        labels = self.clustering_results['K-Means']['labels']
        df_analysis = df.copy()
        df_analysis['cluster'] = labels
        
        segment_profiles = {}
        
        for cluster in sorted(set(labels)):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
                'characteristics': {}
            }
            
            # Calculate key metrics for each cluster
            for feature in feature_cols:
                if feature in df_analysis.columns:
                    profile['characteristics'][feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'median': float(cluster_data[feature].median()),
                        'std': float(cluster_data[feature].std())
                    }
            
            # Determine cluster profile based on key metrics
            avg_spending = profile['characteristics']['annual_spending']['mean']
            avg_frequency = profile['characteristics']['frequency']['mean']
            avg_recency = profile['characteristics']['recency']['mean']
            
            if avg_spending > 3500 and avg_frequency > 30:
                profile['segment_name'] = 'Premium Customers'
                profile['description'] = 'High spending, frequent buyers with strong engagement'
            elif avg_spending > 2000 and avg_frequency > 15:
                profile['segment_name'] = 'Loyal Customers'
                profile['description'] = 'Regular spenders with consistent purchase patterns'
            elif avg_recency > 90 or avg_frequency < 10:
                profile['segment_name'] = 'At-Risk Customers'
                profile['description'] = 'Low engagement, potential churn candidates'
            else:
                profile['segment_name'] = 'Developing Customers'
                profile['description'] = 'Moderate engagement with growth potential'
            
            segment_profiles[f'Cluster {cluster}'] = profile
            
            print(f"\nCluster {cluster} - {profile['segment_name']}:")
            print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
            print(f"  Description: {profile['description']}")
            print(f"  Avg Annual Spending: ${avg_spending:.0f}")
            print(f"  Avg Frequency: {avg_frequency:.1f} purchases")
            print(f"  Avg Recency: {avg_recency:.1f} days")
        
        self.customer_profiles = segment_profiles
        
        # Compare with true segments (if available)
        if 'segment_true' in df.columns:
            print("\n=== Comparison with True Segments ===")
            ari_score = adjusted_rand_score(df['segment_true'], labels)
            print(f"Adjusted Rand Index: {ari_score:.3f}")
            
            # Create confusion matrix
            true_segments = df['segment_true'].unique()
            predicted_clusters = sorted(set(labels))
            
            confusion_data = []
            for true_seg in true_segments:
                true_mask = df['segment_true'] == true_seg
                for cluster in predicted_clusters:
                    cluster_mask = labels == cluster
                    overlap = (true_mask & cluster_mask).sum()
                    confusion_data.append({
                        'True_Segment': true_seg,
                        'Predicted_Cluster': f'Cluster {cluster}',
                        'Count': overlap
                    })
            
            confusion_df = pd.DataFrame(confusion_data)
            confusion_pivot = confusion_df.pivot(index='True_Segment', 
                                               columns='Predicted_Cluster', 
                                               values='Count').fillna(0)
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(confusion_pivot, annot=True, fmt='d', cmap='Blues')
            plt.title('True vs Predicted Segments Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/segment_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        return segment_profiles
    
    def generate_business_recommendations(self):
        """
        Generate actionable business recommendations based on segments
        """
        print("\n=== Business Recommendations ===")
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'segment_strategies': {},
            'general_recommendations': []
        }
        
        for cluster_name, profile in self.customer_profiles.items():
            segment_name = profile['segment_name']
            size = profile['size']
            
            strategies = []
            
            if segment_name == 'Premium Customers':
                strategies.extend([
                    'Implement VIP program with exclusive benefits',
                    'Offer premium products and early access to new releases',
                    'Provide dedicated customer support channel',
                    'Create referral incentives to expand this valuable segment'
                ])
            
            elif segment_name == 'Loyal Customers':
                strategies.extend([
                    'Develop loyalty rewards program to maintain engagement',
                    'Cross-sell complementary products',
                    'Send personalized recommendations based on purchase history',
                    'Gradual upselling to premium tier'
                ])
            
            elif segment_name == 'At-Risk Customers':
                strategies.extend([
                    'Implement win-back email campaigns',
                    'Offer time-limited discounts to re-engage',
                    'Survey to understand reasons for reduced engagement',
                    'Provide customer service outreach for support'
                ])
            
            elif segment_name == 'Developing Customers':
                strategies.extend([
                    'Educational content marketing to increase product awareness',
                    'Progressive discount structure to encourage larger purchases',
                    'Onboarding sequence to improve product adoption',
                    'Social proof and testimonials to build trust'
                ])
            
            recommendations['segment_strategies'][cluster_name] = {
                'segment_name': segment_name,
                'size': size,
                'strategies': strategies
            }
        
        # General recommendations
        total_customers = sum(profile['size'] for profile in self.customer_profiles.values())
        premium_customers = sum(profile['size'] for profile in self.customer_profiles.values() 
                              if profile['segment_name'] == 'Premium Customers')
        at_risk_customers = sum(profile['size'] for profile in self.customer_profiles.values() 
                               if profile['segment_name'] == 'At-Risk Customers')
        
        recommendations['general_recommendations'].extend([
            f"Focus on retaining {premium_customers} premium customers ({premium_customers/total_customers:.1%} of base)",
            f"Urgently address {at_risk_customers} at-risk customers to prevent churn",
            "Implement different communication strategies for each segment",
            "Monitor segment migration over time to track customer lifecycle",
            "Use segment-specific KPIs for marketing campaign measurement",
            "Consider segment-based pricing and promotion strategies"
        ])
        
        # Save recommendations
        with open(f'{self.output_dir}/business_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print("Business Recommendations Generated:")
        for cluster_name, strategy in recommendations['segment_strategies'].items():
            print(f"\n{cluster_name} ({strategy['segment_name']}):")
            for rec in strategy['strategies']:
                print(f"  • {rec}")
        
        print("\nGeneral Recommendations:")
        for rec in recommendations['general_recommendations']:
            print(f"  • {rec}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """
        Run the complete customer segmentation analysis
        """
        print("🚀 Starting Customer Segmentation Analysis")
        print("=" * 60)
        
        # Generate customer dataset
        df = self.generate_customer_dataset()
        
        # Analyze customer features
        feature_cols, correlation_matrix = self.analyze_customer_features(df)
        
        # Prepare data for clustering
        X = df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nData prepared for clustering: {X_scaled.shape[0]} customers, {X_scaled.shape[1]} features")
        
        # Find optimal number of clusters
        elbow_k, silhouette_k, inertias, sil_scores = self.elbow_method_analysis(X_scaled)
        
        # Tune DBSCAN parameters
        best_eps, best_min_samples, dbscan_results = self.dbscan_parameter_tuning(X_scaled)
        
        # Perform clustering with multiple algorithms
        optimal_k = silhouette_k  # Use silhouette-based k for K-means
        clustering_results = self.perform_clustering(X_scaled, optimal_k)
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations(df, X_scaled, feature_cols)
        
        # Analyze customer segments
        segment_profiles = self.analyze_customer_segments(df, feature_cols)
        
        # Generate business recommendations
        recommendations = self.generate_business_recommendations()
        
        print("\n✅ Customer Segmentation Analysis Complete!")
        print("\n📊 Key Insights:")
        print(f"- Identified {optimal_k} distinct customer segments")
        print(f"- Best performing algorithm: K-Means (Silhouette: {self.clustering_results['K-Means']['silhouette_score']:.3f})")
        print(f"- DBSCAN found {self.clustering_results['DBSCAN']['n_clusters']} clusters with {self.clustering_results['DBSCAN']['noise_ratio']:.1%} noise")
        print("- Each segment requires different marketing strategies")
        print("- Focus on retaining premium customers and re-engaging at-risk customers")
        
        return {
            'customer_profiles': segment_profiles,
            'recommendations': recommendations,
            'clustering_performance': self.clustering_results
        }

if __name__ == "__main__":
    # Initialize and run analysis
    segmentation = CustomerSegmentation()
    segmentation.run_complete_analysis()