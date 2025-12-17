"""
Neural Networks From Scratch - Pure NumPy Implementation
Build and train neural networks using only NumPy to understand the fundamentals

"I stopped using libraries and finally understood deep learning."

Author: Your Name
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class ActivationFunctions:
    """
    Collection of activation functions and their derivatives
    """
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of Leaky ReLU function"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        """Softmax activation function for multi-class classification"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class LossFunctions:
    """
    Collection of loss functions and their derivatives
    """
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """Derivative of MSE loss"""
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """Binary cross-entropy loss"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        """Derivative of binary cross-entropy loss"""
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """Categorical cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        """Derivative of categorical cross-entropy loss"""
        return (y_pred - y_true) / y_true.shape[0]

class NeuralNetwork:
    """
    Neural Network implementation from scratch using only NumPy
    """
    
    def __init__(self, layers, activation='relu', output_activation='sigmoid', 
                 loss='binary_crossentropy', learning_rate=0.01, random_state=42):
        """
        Initialize neural network
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            loss: Loss function to use
            learning_rate: Learning rate for gradient descent
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)
        
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_name = loss
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Xavier/Glorot initialization for better convergence
        for i in range(self.num_layers - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            b = np.zeros((1, layers[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation functions
        self.activation_func = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(ActivationFunctions, f"{activation}_derivative")
        self.output_activation_func = getattr(ActivationFunctions, output_activation)
        
        if output_activation == 'softmax':
            self.output_activation_derivative = None  # Handled differently for softmax
        else:
            self.output_activation_derivative = getattr(ActivationFunctions, f"{output_activation}_derivative")
        
        # Set loss function
        self.loss_func = getattr(LossFunctions, loss)
        self.loss_derivative = getattr(LossFunctions, f"{loss}_derivative")
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        """
        activations = [X]
        z_values = []
        
        # Forward pass through hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.activation_func(z)
            activations.append(a)
        
        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        
        if self.output_activation_name == 'softmax':
            a = self.output_activation_func(z)
        else:
            a = self.output_activation_func(z)
        
        activations.append(a)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        # Calculate output layer error
        if self.output_activation_name == 'softmax' and self.loss_name == 'categorical_crossentropy':
            # For softmax + categorical crossentropy, derivative simplifies
            delta = activations[-1] - y
        else:
            # General case
            loss_grad = self.loss_derivative(y, activations[-1])
            if self.output_activation_derivative:
                output_grad = self.output_activation_derivative(z_values[-1])
                delta = loss_grad * output_grad
            else:
                delta = loss_grad
        
        # Output layer gradients
        dW[-1] = np.dot(activations[-2].T, delta)
        db[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(z_values[i])
            dW[i] = np.dot(activations[i].T, delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Update weights and biases using gradient descent
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def predict(self, X):
        """
        Make predictions
        """
        probabilities = self.predict_proba(X)
        if probabilities.shape[1] == 1:
            # Binary classification
            return (probabilities > 0.5).astype(int)
        else:
            # Multi-class classification
            return np.argmax(probabilities, axis=1)
    
    def calculate_accuracy(self, X, y):
        """
        Calculate accuracy
        """
        predictions = self.predict(X)
        if len(y.shape) > 1 and y.shape[1] > 1:
            # One-hot encoded labels
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.flatten()
        
        return np.mean(predictions.flatten() == y_true)
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward pass
                activations, z_values = self.forward_propagation(X_batch)
                
                # Calculate loss
                batch_loss = self.loss_func(y_batch, activations[-1])
                epoch_loss += batch_loss
                
                # Backward pass
                dW, db = self.backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            # Calculate metrics
            train_loss = epoch_loss / (n_samples // batch_size + 1)
            train_accuracy = self.calculate_accuracy(X, y)
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.predict_proba(X_val)
                val_loss = self.loss_func(y_val, val_predictions)
                val_accuracy = self.calculate_accuracy(X_val, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                if X_val is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                print()

class NeuralNetworkExperiments:
    """
    Comprehensive experiments with neural networks from scratch
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_datasets(self):
        """
        Generate different types of datasets for testing
        """
        print("Generating datasets for neural network experiments...")
        
        datasets = {}
        
        # 1. Linear classification (easy)
        X_linear, y_linear = make_classification(
            n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )
        datasets['linear'] = (X_linear, y_linear, "Linear Classification")
        
        # 2. Non-linear classification (moons dataset)
        X_moons, y_moons = make_moons(n_samples=1000, noise=0.2, random_state=42)
        datasets['moons'] = (X_moons, y_moons, "Moons Classification")
        
        # 3. Multi-class classification (digits)
        digits = load_digits()
        X_digits = digits.data / 16.0  # Normalize pixel values
        y_digits = digits.target
        datasets['digits'] = (X_digits, y_digits, "Digits Classification")
        
        # 4. Complex non-linear (concentric circles)
        X_circles, y_circles = make_classification(
            n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=2, random_state=42
        )
        # Make it more circular
        r = np.sqrt(X_circles[:, 0]**2 + X_circles[:, 1]**2)
        theta = np.arctan2(X_circles[:, 1], X_circles[:, 0])
        X_circles = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        datasets['circles'] = (X_circles, y_circles, "Circles Classification")
        
        print(f"Generated {len(datasets)} datasets for experiments")
        
        return datasets
    
    def experiment_activation_functions(self, datasets):
        """
        Compare different activation functions
        """
        print("\n=== Activation Functions Comparison ===")
        
        activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
        results = {}
        
        # Use moons dataset for this experiment
        X, y, _ = datasets['moons']
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Reshape for neural network
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        for activation in activation_functions:
            print(f"Training with {activation} activation...")
            
            # Create network
            nn = NeuralNetwork(
                layers=[2, 10, 8, 1],
                activation=activation,
                output_activation='sigmoid',
                loss='binary_crossentropy',
                learning_rate=0.01,
                random_state=42
            )
            
            # Train
            nn.fit(X_train, y_train, X_test, y_test, epochs=500, verbose=False)
            
            # Evaluate
            train_acc = nn.calculate_accuracy(X_train, y_train)
            test_acc = nn.calculate_accuracy(X_test, y_test)
            
            results[activation] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'final_train_loss': nn.history['loss'][-1],
                'final_val_loss': nn.history['val_loss'][-1],
                'history': nn.history
            }
            
            print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        self.results['activation_comparison'] = results
        return results
    
    def experiment_network_architectures(self, datasets):
        """
        Compare different network architectures
        """
        print("\n=== Network Architecture Comparison ===")
        
        architectures = [
            ([2, 5, 1], "Small Network"),
            ([2, 10, 8, 1], "Medium Network"),
            ([2, 20, 15, 10, 1], "Large Network"),
            ([2, 50, 25, 10, 1], "Very Large Network")
        ]
        
        results = {}
        
        # Use moons dataset
        X, y, _ = datasets['moons']
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        for layers, name in architectures:
            print(f"Training {name}: {layers}...")
            
            nn = NeuralNetwork(
                layers=layers,
                activation='relu',
                output_activation='sigmoid',
                loss='binary_crossentropy',
                learning_rate=0.01,
                random_state=42
            )
            
            nn.fit(X_train, y_train, X_test, y_test, epochs=500, verbose=False)
            
            train_acc = nn.calculate_accuracy(X_train, y_train)
            test_acc = nn.calculate_accuracy(X_test, y_test)
            
            # Count parameters
            total_params = sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
            
            results[name] = {
                'layers': layers,
                'parameters': total_params,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'final_train_loss': nn.history['loss'][-1],
                'final_val_loss': nn.history['val_loss'][-1],
                'history': nn.history
            }
            
            print(f"  Params: {total_params}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        self.results['architecture_comparison'] = results
        return results
    
    def experiment_multiclass_classification(self, datasets):
        """
        Test on multi-class classification (digits)
        """
        print("\n=== Multi-class Classification (Digits) ===")
        
        X, y, _ = datasets['digits']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # One-hot encode labels
        num_classes = len(np.unique(y))
        y_train_onehot = np.eye(num_classes)[y_train]
        y_test_onehot = np.eye(num_classes)[y_test]
        
        print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features, {num_classes} classes")
        
        # Create network for multi-class classification
        nn = NeuralNetwork(
            layers=[X_train.shape[1], 128, 64, num_classes],
            activation='relu',
            output_activation='softmax',
            loss='categorical_crossentropy',
            learning_rate=0.01,
            random_state=42
        )
        
        # Train
        nn.fit(X_train, y_train_onehot, X_test, y_test_onehot, 
               epochs=1000, batch_size=32, verbose=True)
        
        # Evaluate
        train_acc = nn.calculate_accuracy(X_train, y_train_onehot)
        test_acc = nn.calculate_accuracy(X_test, y_test_onehot)
        
        print(f"Final Results - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        self.results['multiclass_digits'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'num_classes': num_classes,
            'history': nn.history,
            'model': nn
        }
        
        return nn, test_acc
    
    def create_comprehensive_visualizations(self, datasets):
        """
        Create comprehensive visualizations of all experiments
        """
        print("\n=== Creating Comprehensive Visualizations ===")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # 1. Dataset visualizations
        dataset_names = ['linear', 'moons', 'circles', 'digits']
        for i, name in enumerate(dataset_names):
            if name in datasets:
                X, y, title = datasets[name]
                
                if name == 'digits':
                    # Show some digit examples
                    digit_images = X[:16].reshape(-1, 8, 8)
                    for j in range(4):
                        for k in range(4):
                            idx = j * 4 + k
                            axes[0, i].imshow(digit_images[idx], cmap='gray')
                            axes[0, i].set_title(f'Digit: {y[idx]}' if j == 0 and k == 0 else '')
                            axes[0, i].axis('off')
                    break
                else:
                    # 2D scatter plot
                    scatter = axes[0, i].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
                    axes[0, i].set_title(title)
                    plt.colorbar(scatter, ax=axes[0, i])
        
        # 2. Activation function comparison
        if 'activation_comparison' in self.results:
            activations = list(self.results['activation_comparison'].keys())
            test_accs = [self.results['activation_comparison'][act]['test_accuracy'] 
                        for act in activations]
            
            bars = axes[1, 0].bar(activations, test_accs, color='lightblue')
            axes[1, 0].set_title('Activation Functions Comparison')
            axes[1, 0].set_ylabel('Test Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, acc in zip(bars, test_accs):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Architecture comparison
        if 'architecture_comparison' in self.results:
            arch_results = self.results['architecture_comparison']
            names = list(arch_results.keys())
            params = [arch_results[name]['parameters'] for name in names]
            test_accs = [arch_results[name]['test_accuracy'] for name in names]
            
            # Parameters vs Accuracy
            axes[1, 1].scatter(params, test_accs, s=100, alpha=0.7)
            axes[1, 1].set_xlabel('Number of Parameters')
            axes[1, 1].set_ylabel('Test Accuracy')
            axes[1, 1].set_title('Architecture Size vs Performance')
            
            for i, name in enumerate(names):
                axes[1, 1].annotate(name.replace(' Network', ''), 
                                   (params[i], test_accs[i]),
                                   xytext=(5, 5), textcoords='offset points')
        
        # 4. Training curves for best activation
        if 'activation_comparison' in self.results:
            best_activation = max(self.results['activation_comparison'].keys(),
                                key=lambda x: self.results['activation_comparison'][x]['test_accuracy'])
            
            history = self.results['activation_comparison'][best_activation]['history']
            
            axes[1, 2].plot(history['loss'], label='Training Loss', color='blue')
            axes[1, 2].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].set_title(f'Training Curves ({best_activation})')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 5. Accuracy curves for best activation
        if 'activation_comparison' in self.results:
            history = self.results['activation_comparison'][best_activation]['history']
            
            axes[1, 3].plot(history['accuracy'], label='Training Accuracy', color='blue')
            axes[1, 3].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
            axes[1, 3].set_xlabel('Epoch')
            axes[1, 3].set_ylabel('Accuracy')
            axes[1, 3].set_title(f'Accuracy Curves ({best_activation})')
            axes[1, 3].legend()
            axes[1, 3].grid(True, alpha=0.3)
        
        # 6. Multi-class confusion matrix (if available)
        if 'multiclass_digits' in self.results:
            # Create a sample confusion matrix visualization
            from sklearn.metrics import confusion_matrix
            
            # Get model and test data
            model = self.results['multiclass_digits']['model']
            X, y, _ = datasets['digits']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
            axes[2, 0].set_title('Digits Classification Confusion Matrix')
            axes[2, 0].set_xlabel('Predicted')
            axes[2, 0].set_ylabel('Actual')
        
        # 7. Learning rate comparison (demonstration)
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        final_losses = []
        
        X, y, _ = datasets['moons']
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.reshape(-1, 1)
        
        for lr in learning_rates:
            nn = NeuralNetwork([2, 10, 1], learning_rate=lr, random_state=42)
            nn.fit(X_train, y_train, epochs=200, verbose=False)
            final_losses.append(nn.history['loss'][-1])
        
        axes[2, 1].semilogx(learning_rates, final_losses, 'bo-')
        axes[2, 1].set_xlabel('Learning Rate')
        axes[2, 1].set_ylabel('Final Training Loss')
        axes[2, 1].set_title('Learning Rate Impact')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 8. Activation functions shapes
        x = np.linspace(-5, 5, 100)
        
        axes[2, 2].plot(x, ActivationFunctions.sigmoid(x), label='Sigmoid')
        axes[2, 2].plot(x, ActivationFunctions.tanh(x), label='Tanh')
        axes[2, 2].plot(x, ActivationFunctions.relu(x), label='ReLU')
        axes[2, 2].plot(x, ActivationFunctions.leaky_relu(x), label='Leaky ReLU')
        
        axes[2, 2].set_xlabel('Input')
        axes[2, 2].set_ylabel('Output')
        axes[2, 2].set_title('Activation Functions')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # 9. Network architecture visualization (conceptual)
        # Simple representation of a neural network
        layer_sizes = [4, 6, 4, 2]
        layer_positions = [i for i in range(len(layer_sizes))]
        
        for i, size in enumerate(layer_sizes):
            y_positions = np.linspace(-size/2, size/2, size)
            axes[2, 3].scatter([i] * size, y_positions, s=100, 
                              c='lightblue' if i < len(layer_sizes)-1 else 'lightcoral')
            
            # Draw connections
            if i < len(layer_sizes) - 1:
                next_y_positions = np.linspace(-layer_sizes[i+1]/2, layer_sizes[i+1]/2, layer_sizes[i+1])
                for y1 in y_positions:
                    for y2 in next_y_positions:
                        axes[2, 3].plot([i, i+1], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
        
        axes[2, 3].set_title('Neural Network Architecture')
        axes[2, 3].set_xlabel('Layers')
        axes[2, 3].set_ylabel('Neurons')
        axes[2, 3].set_xticks(range(len(layer_sizes)))
        axes[2, 3].set_xticklabels(['Input', 'Hidden 1', 'Hidden 2', 'Output'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/neural_networks_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_results(self):
        """
        Save detailed experimental results
        """
        print("\n=== Saving Detailed Results ===")
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for experiment, results in self.results.items():
            serializable_results[experiment] = {}
            
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[experiment][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_results[experiment][key] = float(value)
                elif key == 'model':
                    # Skip model objects
                    continue
                else:
                    serializable_results[experiment][key] = value
        
        # Add summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'experiments_conducted': list(self.results.keys()),
            'key_insights': [
                "ReLU activation generally performs best for hidden layers",
                "Deeper networks can capture more complex patterns but may overfit",
                "Learning rate significantly impacts convergence speed and final performance",
                "Multi-class classification requires softmax output and categorical cross-entropy",
                "Proper initialization is crucial for network convergence"
            ],
            'best_configurations': {}
        }
        
        # Find best configurations
        if 'activation_comparison' in self.results:
            best_activation = max(self.results['activation_comparison'].keys(),
                                key=lambda x: self.results['activation_comparison'][x]['test_accuracy'])
            summary['best_configurations']['activation'] = best_activation
        
        if 'architecture_comparison' in self.results:
            best_arch = max(self.results['architecture_comparison'].keys(),
                          key=lambda x: self.results['architecture_comparison'][x]['test_accuracy'])
            summary['best_configurations']['architecture'] = best_arch
        
        # Save results
        results_data = {
            'summary': summary,
            'detailed_results': serializable_results
        }
        
        with open(f'{self.output_dir}/neural_network_experiments.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/neural_network_experiments.json")
        
        return results_data
    
    def run_complete_analysis(self):
        """
        Run all neural network experiments
        """
        print("🚀 Starting Neural Networks from Scratch Analysis")
        print("=" * 60)
        
        # Generate datasets
        datasets = self.generate_datasets()
        
        # Run experiments
        print("\n" + "="*50)
        print("PHASE 1: ACTIVATION FUNCTION ANALYSIS")
        print("="*50)
        activation_results = self.experiment_activation_functions(datasets)
        
        print("\n" + "="*50)
        print("PHASE 2: NETWORK ARCHITECTURE ANALYSIS")
        print("="*50)
        architecture_results = self.experiment_network_architectures(datasets)
        
        print("\n" + "="*50)
        print("PHASE 3: MULTI-CLASS CLASSIFICATION")
        print("="*50)
        multiclass_model, multiclass_acc = self.experiment_multiclass_classification(datasets)
        
        print("\n" + "="*50)
        print("PHASE 4: VISUALIZATION AND ANALYSIS")
        print("="*50)
        self.create_comprehensive_visualizations(datasets)
        
        # Save results
        results_data = self.save_detailed_results()
        
        print("\n✅ Neural Networks from Scratch Analysis Complete!")
        print("\n📊 Key Findings:")
        
        if 'activation_comparison' in self.results:
            best_activation = max(self.results['activation_comparison'].keys(),
                                key=lambda x: self.results['activation_comparison'][x]['test_accuracy'])
            best_acc = self.results['activation_comparison'][best_activation]['test_accuracy']
            print(f"- Best activation function: {best_activation} (accuracy: {best_acc:.4f})")
        
        if 'architecture_comparison' in self.results:
            best_arch = max(self.results['architecture_comparison'].keys(),
                          key=lambda x: self.results['architecture_comparison'][x]['test_accuracy'])
            best_params = self.results['architecture_comparison'][best_arch]['parameters']
            best_arch_acc = self.results['architecture_comparison'][best_arch]['test_accuracy']
            print(f"- Best architecture: {best_arch} ({best_params} params, accuracy: {best_arch_acc:.4f})")
        
        if 'multiclass_digits' in self.results:
            digits_acc = self.results['multiclass_digits']['test_accuracy']
            print(f"- Multi-class digits accuracy: {digits_acc:.4f}")
        
        print("- Neural networks can learn complex non-linear patterns")
        print("- Proper activation functions and architectures are crucial")
        print("- Understanding backpropagation enables effective debugging")
        
        return results_data

if __name__ == "__main__":
    # Initialize and run experiments
    experiments = NeuralNetworkExperiments()
    experiments.run_complete_analysis()