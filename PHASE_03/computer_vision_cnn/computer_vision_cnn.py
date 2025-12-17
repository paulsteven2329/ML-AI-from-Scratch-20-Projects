"""
Computer Vision with Convolutional Neural Networks
Demonstrates CNNs, convolutions, pooling, and transfer learning for image classification

"Why CNNs see better than humans (sometimes)."

Author: Your Name  
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import cv2
import warnings

warnings.filterwarnings('ignore')

class ConvolutionalNeuralNetwork:
    """
    Comprehensive CNN implementation for computer vision tasks
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.models = {}
        self.histories = {}
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_synthetic_image_data(self, n_samples=2000, img_size=(64, 64)):
        """
        Create synthetic image data for demonstration
        """
        print(f"Creating synthetic image dataset: {n_samples} samples of size {img_size}")
        
        width, height = img_size
        X = np.zeros((n_samples, height, width, 3))
        y = np.zeros(n_samples)
        
        # Create 4 classes of synthetic images
        samples_per_class = n_samples // 4
        
        for i in range(n_samples):
            class_idx = i // samples_per_class
            class_idx = min(class_idx, 3)  # Ensure we don't exceed 4 classes
            
            img = np.zeros((height, width, 3))
            
            if class_idx == 0:  # Geometric shapes - Circles
                center_x = np.random.randint(width//4, 3*width//4)
                center_y = np.random.randint(height//4, 3*height//4)
                radius = np.random.randint(width//8, width//4)
                color = np.random.rand(3)
                
                y_coords, x_coords = np.ogrid[:height, :width]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                img[mask] = color
                
            elif class_idx == 1:  # Geometric shapes - Rectangles
                x1 = np.random.randint(0, width//2)
                y1 = np.random.randint(0, height//2)
                x2 = np.random.randint(width//2, width)
                y2 = np.random.randint(height//2, height)
                color = np.random.rand(3)
                
                img[y1:y2, x1:x2] = color
                
            elif class_idx == 2:  # Noise patterns
                noise_level = np.random.uniform(0.3, 0.8)
                img = np.random.rand(height, width, 3) * noise_level
                
                # Add some structure
                for _ in range(5):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    size = np.random.randint(5, 15)
                    img[max(0, y-size):min(height, y+size), 
                        max(0, x-size):min(width, x+size)] = np.random.rand(3)
                
            else:  # Gradient patterns
                direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
                colors = np.random.rand(2, 3)
                
                if direction == 'horizontal':
                    for x in range(width):
                        ratio = x / width
                        img[:, x] = colors[0] * (1 - ratio) + colors[1] * ratio
                elif direction == 'vertical':
                    for y in range(height):
                        ratio = y / height
                        img[y, :] = colors[0] * (1 - ratio) + colors[1] * ratio
                else:  # diagonal
                    for x in range(width):
                        for y in range(height):
                            ratio = (x + y) / (width + height)
                            img[y, x] = colors[0] * (1 - ratio) + colors[1] * ratio
            
            X[i] = img
            y[i] = class_idx
        
        # Normalize pixel values
        X = X.astype(np.float32)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, X.shape)
        X = np.clip(X + noise, 0, 1)
        
        print(f"Dataset created: {X.shape[0]} images of shape {X.shape[1:]}")
        print(f"Classes distribution: {np.bincount(y.astype(int))}")
        
        return X, y
    
    def create_simple_cnn(self, input_shape, num_classes):
        """
        Create a simple CNN architecture
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_advanced_cnn(self, input_shape, num_classes):
        """
        Create a more advanced CNN with batch normalization and more layers
        """
        model = keras.Sequential([
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self, input_shape, num_classes, base_model_name='VGG16'):
        """
        Create a model using transfer learning
        """
        # Load pre-trained model
        if base_model_name == 'VGG16':
            base_model = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'ResNet50':
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'MobileNetV2':
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classifier
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def visualize_convolution_operation(self, X_sample):
        """
        Visualize how convolution operation works
        """
        print("\n=== Visualizing Convolution Operation ===")
        
        # Take a sample image
        sample_img = X_sample[0] if len(X_sample.shape) == 4 else X_sample
        if sample_img.shape[-1] == 3:
            # Convert to grayscale for clearer visualization
            sample_img = np.dot(sample_img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Define different kernels
        kernels = {
            'Edge Detection (Horizontal)': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            'Edge Detection (Vertical)': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(sample_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Apply different kernels
        for i, (kernel_name, kernel) in enumerate(kernels.items()):
            if i >= 5:  # We only have 5 more subplot positions
                break
            
            # Apply convolution
            filtered_img = cv2.filter2D(sample_img, -1, kernel)
            
            row = i // 2 if i < 2 else 1
            col = (i + 1) % 3 if i < 2 else i - 1
            
            axes[row, col].imshow(filtered_img, cmap='gray')
            axes[row, col].set_title(f'{kernel_name}\nKernel: {kernel.shape}')
            axes[row, col].axis('off')
        
        # Show kernel visualization
        kernel_to_show = kernels['Edge Detection (Horizontal)']
        axes[1, 2].imshow(kernel_to_show, cmap='RdBu', interpolation='nearest')
        axes[1, 2].set_title('Sample Kernel\n(Edge Detection)')
        
        # Add grid and values
        for i in range(kernel_to_show.shape[0]):
            for j in range(kernel_to_show.shape[1]):
                axes[1, 2].text(j, i, f'{kernel_to_show[i, j]:.0f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/convolution_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_and_compare_models(self, X, y):
        """
        Train and compare different CNN architectures
        """
        print("\n=== Training and Comparing CNN Models ===")
        
        # Prepare data
        num_classes = len(np.unique(y))
        y_categorical = keras.utils.to_categorical(y, num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train.argmax(axis=1)
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        input_shape = X_train.shape[1:]
        
        # Create models
        models_to_train = {
            'Simple CNN': self.create_simple_cnn(input_shape, num_classes),
            'Advanced CNN': self.create_advanced_cnn(input_shape, num_classes)
        }
        
        # Add transfer learning model (resize images if needed)
        if input_shape[0] >= 32 and input_shape[1] >= 32:
            # Resize for transfer learning (most pre-trained models expect larger images)
            target_size = (64, 64, 3) if input_shape[2] == 3 else (64, 64, 1)
            
            X_train_resized = tf.image.resize(X_train, [64, 64]).numpy()
            X_val_resized = tf.image.resize(X_val, [64, 64]).numpy()
            X_test_resized = tf.image.resize(X_test, [64, 64]).numpy()
            
            # Ensure 3 channels for transfer learning
            if target_size[2] == 3 and X_train_resized.shape[-1] == 1:
                X_train_resized = np.repeat(X_train_resized, 3, axis=-1)
                X_val_resized = np.repeat(X_val_resized, 3, axis=-1)
                X_test_resized = np.repeat(X_test_resized, 3, axis=-1)
            
            transfer_model, base_model = self.create_transfer_learning_model(
                (64, 64, 3), num_classes, 'MobileNetV2'
            )
            models_to_train['Transfer Learning (MobileNetV2)'] = transfer_model
        
        # Training parameters
        epochs = 50
        batch_size = 32
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7
        )
        
        # Train models
        for model_name, model in models_to_train.items():
            print(f"\nTraining {model_name}...")
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Print model summary
            print(f"Model parameters: {model.count_params():,}")
            
            # Prepare data for this model
            if 'Transfer Learning' in model_name:
                X_train_model = X_train_resized
                X_val_model = X_val_resized
                X_test_model = X_test_resized
            else:
                X_train_model = X_train
                X_val_model = X_val
                X_test_model = X_test
            
            # Data augmentation for better generalization
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # Train
            history = model.fit(
                datagen.flow(X_train_model, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train_model) // batch_size,
                epochs=epochs,
                validation_data=(X_val_model, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test_model, y_test, verbose=0)
            
            # Store results
            self.models[model_name] = model
            self.histories[model_name] = history.history
            self.results[model_name] = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'parameters': model.count_params(),
                'final_epoch': len(history.history['loss'])
            }
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return X_test, y_test, X_test_resized if 'Transfer Learning (MobileNetV2)' in models_to_train else None
    
    def visualize_feature_maps(self, model, X_sample, layer_indices=[0, 2, 4]):
        """
        Visualize feature maps from different layers
        """
        print("\n=== Visualizing Feature Maps ===")
        
        # Get intermediate outputs
        layer_outputs = [model.layers[i].output for i in layer_indices if i < len(model.layers)]
        
        if not layer_outputs:
            print("No valid layers found for visualization")
            return
        
        activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations for a sample image
        sample_input = X_sample[:1] if len(X_sample.shape) == 4 else X_sample[np.newaxis, ...]
        activations = activation_model.predict(sample_input)
        
        if not isinstance(activations, list):
            activations = [activations]
        
        # Create visualization
        fig, axes = plt.subplots(len(activations), 8, figsize=(20, 3 * len(activations)))
        
        if len(activations) == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, activation in enumerate(activations):
            layer_name = model.layers[layer_indices[layer_idx]].name
            
            # Show first 8 feature maps
            for i in range(min(8, activation.shape[-1])):
                feature_map = activation[0, :, :, i]
                
                axes[layer_idx, i].imshow(feature_map, cmap='viridis')
                axes[layer_idx, i].set_title(f'{layer_name}\nFeature {i+1}')
                axes[layer_idx, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_maps_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_analysis(self, X_test, y_test, X_test_resized=None):
        """
        Create comprehensive analysis and visualizations
        """
        print("\n=== Creating Comprehensive Analysis ===")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. Model Performance Comparison
        model_names = list(self.results.keys())
        test_accuracies = [self.results[name]['test_accuracy'] for name in model_names]
        parameters = [self.results[name]['parameters'] for name in model_names]
        
        bars = axes[0, 0].bar(range(len(model_names)), test_accuracies, color='lightblue')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in model_names])
        
        # Add value labels
        for bar, acc in zip(bars, test_accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Parameters vs Performance
        axes[0, 1].scatter(parameters, test_accuracies, s=100, alpha=0.7, c=['red', 'green', 'blue'])
        axes[0, 1].set_xlabel('Model Parameters')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Parameters vs Performance')
        
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name.split()[0], (parameters[i], test_accuracies[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        # 3. Training History - Loss
        for model_name in model_names:
            if model_name in self.histories:
                history = self.histories[model_name]
                epochs = range(1, len(history['loss']) + 1)
                axes[0, 2].plot(epochs, history['loss'], label=f'{model_name} - Train')
                axes[0, 2].plot(epochs, history['val_loss'], '--', label=f'{model_name} - Val')
        
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss Curves')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training History - Accuracy
        for model_name in model_names:
            if model_name in self.histories:
                history = self.histories[model_name]
                epochs = range(1, len(history['accuracy']) + 1)
                axes[1, 0].plot(epochs, history['accuracy'], label=f'{model_name} - Train')
                axes[1, 0].plot(epochs, history['val_accuracy'], '--', label=f'{model_name} - Val')
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy Curves')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Sample predictions visualization
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.models[best_model_name]
        
        # Prepare test data for the best model
        if 'Transfer Learning' in best_model_name and X_test_resized is not None:
            X_test_model = X_test_resized
        else:
            X_test_model = X_test
        
        predictions = best_model.predict(X_test_model)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Show some sample predictions
        sample_indices = np.random.choice(len(X_test_model), 6, replace=False)
        
        for i, idx in enumerate(sample_indices):
            row = 1 + i // 3
            col = 1 + i % 3
            if row < 3 and col < 3:
                axes[row, col].imshow(X_test_model[idx])
                axes[row, col].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}')
                axes[row, col].axis('off')
        
        # 6. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 2])
        axes[2, 2].set_title(f'Confusion Matrix\n({best_model_name})')
        axes[2, 2].set_xlabel('Predicted')
        axes[2, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cnn_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_pooling_operations(self, X_sample):
        """
        Demonstrate different pooling operations
        """
        print("\n=== Demonstrating Pooling Operations ===")
        
        # Take a sample image
        sample_img = X_sample[0] if len(X_sample.shape) == 4 else X_sample
        if len(sample_img.shape) == 3:
            sample_img = sample_img[:, :, 0]  # Take one channel
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(sample_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Max pooling 2x2
        max_pool_2x2 = tf.nn.max_pool2d(
            sample_img[np.newaxis, :, :, np.newaxis],
            ksize=2, strides=2, padding='VALID'
        )[0, :, :, 0]
        
        axes[0, 1].imshow(max_pool_2x2, cmap='gray')
        axes[0, 1].set_title('Max Pooling 2x2')
        axes[0, 1].axis('off')
        
        # Average pooling 2x2
        avg_pool_2x2 = tf.nn.avg_pool2d(
            sample_img[np.newaxis, :, :, np.newaxis],
            ksize=2, strides=2, padding='VALID'
        )[0, :, :, 0]
        
        axes[0, 2].imshow(avg_pool_2x2, cmap='gray')
        axes[0, 2].set_title('Average Pooling 2x2')
        axes[0, 2].axis('off')
        
        # Max pooling 4x4
        max_pool_4x4 = tf.nn.max_pool2d(
            sample_img[np.newaxis, :, :, np.newaxis],
            ksize=4, strides=4, padding='VALID'
        )[0, :, :, 0]
        
        axes[1, 0].imshow(max_pool_4x4, cmap='gray')
        axes[1, 0].set_title('Max Pooling 4x4')
        axes[1, 0].axis('off')
        
        # Show pooling effect on image dimensions
        sizes = [sample_img.shape, max_pool_2x2.shape, avg_pool_2x2.shape, max_pool_4x4.shape]
        size_labels = ['Original', 'Max Pool 2x2', 'Avg Pool 2x2', 'Max Pool 4x4']
        
        bars = axes[1, 1].bar(range(len(sizes)), [s[0] * s[1] for s in sizes], color='lightgreen')
        axes[1, 1].set_title('Image Size After Pooling')
        axes[1, 1].set_ylabel('Total Pixels')
        axes[1, 1].set_xticks(range(len(sizes)))
        axes[1, 1].set_xticklabels(size_labels, rotation=45)
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = size[0] * size[1]
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 50,
                           f'{size[0]}x{size[1]}', ha='center', va='bottom')
        
        # Pooling comparison
        pooling_data = {
            'Operation': size_labels,
            'Height': [s[0] for s in sizes],
            'Width': [s[1] for s in sizes],
            'Total Pixels': [s[0] * s[1] for s in sizes]
        }
        
        # Create table
        table_data = [[pooling_data['Operation'][i], 
                      f"{pooling_data['Height'][i]}x{pooling_data['Width'][i]}",
                      pooling_data['Total Pixels'][i]] for i in range(len(sizes))]
        
        table = axes[1, 2].table(cellText=table_data, 
                                colLabels=['Operation', 'Dimensions', 'Total Pixels'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Pooling Operations Summary')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pooling_operations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_comprehensive_results(self):
        """
        Save comprehensive results and insights
        """
        print("\n=== Saving Comprehensive Results ===")
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'test_accuracy': float(result['test_accuracy']),
                'test_loss': float(result['test_loss']),
                'parameters': int(result['parameters']),
                'final_epoch': int(result['final_epoch'])
            }
        
        # Add training histories
        serializable_histories = {}
        for model_name, history in self.histories.items():
            serializable_histories[model_name] = {
                key: [float(val) for val in values] 
                for key, values in history.items()
            }
        
        # Create comprehensive summary
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'models_trained': len(self.results),
                'best_model': best_model_name,
                'best_accuracy': float(self.results[best_model_name]['test_accuracy']),
                'total_parameters_tested': sum(result['parameters'] for result in self.results.values())
            },
            'model_results': serializable_results,
            'training_histories': serializable_histories,
            'key_insights': [
                "CNNs are highly effective for image classification tasks",
                "Transfer learning can achieve excellent results with fewer parameters",
                "Data augmentation helps improve generalization",
                "Batch normalization and dropout reduce overfitting",
                "Feature maps show hierarchical pattern recognition",
                "Pooling operations reduce spatial dimensions while preserving important features"
            ],
            'architectural_lessons': [
                "Deeper networks can capture more complex patterns",
                "Convolutional layers extract spatial features effectively", 
                "Max pooling preserves strong features while reducing computation",
                "Global average pooling can replace flatten + dense layers",
                "Pre-trained models provide excellent feature extractors"
            ]
        }
        
        # Save results
        with open(f'{self.output_dir}/cnn_comprehensive_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/cnn_comprehensive_results.json")
        
        return comprehensive_results
    
    def run_complete_analysis(self):
        """
        Run the complete CNN analysis
        """
        print("🚀 Starting Computer Vision with CNNs Analysis")
        print("=" * 60)
        
        # Generate synthetic image data
        X, y = self.create_synthetic_image_data(n_samples=2000, img_size=(64, 64))
        
        print("\n" + "="*50)
        print("PHASE 1: CONVOLUTION OPERATION VISUALIZATION")
        print("="*50)
        self.visualize_convolution_operation(X)
        
        print("\n" + "="*50)
        print("PHASE 2: POOLING OPERATIONS DEMONSTRATION")
        print("="*50)
        self.demonstrate_pooling_operations(X)
        
        print("\n" + "="*50)
        print("PHASE 3: CNN MODEL TRAINING AND COMPARISON")
        print("="*50)
        X_test, y_test, X_test_resized = self.train_and_compare_models(X, y)
        
        print("\n" + "="*50)
        print("PHASE 4: FEATURE VISUALIZATION")
        print("="*50)
        # Visualize feature maps for the best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.models[best_model_name]
        
        # Find convolutional layers
        conv_layer_indices = []
        for i, layer in enumerate(best_model.layers):
            if isinstance(layer, layers.Conv2D):
                conv_layer_indices.append(i)
        
        if conv_layer_indices:
            sample_data = X_test_resized if 'Transfer Learning' in best_model_name and X_test_resized is not None else X_test
            self.visualize_feature_maps(best_model, sample_data[0], conv_layer_indices[:3])
        
        print("\n" + "="*50)
        print("PHASE 5: COMPREHENSIVE ANALYSIS")
        print("="*50)
        self.create_comprehensive_analysis(X_test, y_test, X_test_resized)
        
        # Save results
        results = self.save_comprehensive_results()
        
        print("\n✅ Computer Vision with CNNs Analysis Complete!")
        print("\n📊 Key Results:")
        
        for model_name, result in self.results.items():
            print(f"- {model_name}: {result['test_accuracy']:.4f} accuracy ({result['parameters']:,} parameters)")
        
        print(f"\n🏆 Best Model: {results['summary']['best_model']}")
        print(f"🎯 Best Accuracy: {results['summary']['best_accuracy']:.4f}")
        print("\n💡 Key Insights:")
        print("- CNNs automatically learn hierarchical feature representations")
        print("- Convolutional layers detect local patterns (edges, textures)")
        print("- Pooling layers provide translation invariance and reduce computation")
        print("- Transfer learning leverages pre-trained features for new tasks")
        print("- Data augmentation improves model generalization")
        
        return results

if __name__ == "__main__":
    # Initialize and run analysis
    cnn_analysis = ConvolutionalNeuralNetwork()
    cnn_analysis.run_complete_analysis()