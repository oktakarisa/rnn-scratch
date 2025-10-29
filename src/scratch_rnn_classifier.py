import numpy as np
from src.simple_rnn import SimpleRNN


class ScratchSimpleRNNClassifier:
    """
    RNN Classifier that follows the assignment requirements.
    
    This class extends SimpleRNN to create a complete classifier as requested
    in the assignment, following the structure of ScratchDeepNeuralNetrowkClassifier
    from the previous Sprint.
    """
    
    def __init__(self, n_nodes, n_features, n_output=1, activation='tanh', 
                 lr=0.01, batch_size=32, epochs=10, verbose=True, random_state=None):
        """
        Initialize the RNN classifier.
        
        Parameters
        ----------
        n_nodes : int
            Number of RNN nodes
        n_features : int
            Number of input features
        n_output : int, optional
            Number of output classes. Default is 1.
        activation : str, optional
            Activation function. Default is 'tanh'.
        lr : float, optional
            Learning rate. Default is 0.01.
        batch_size : int, optional
            Batch size. Default is 32.
        epochs : int, optional
            Number of epochs. Default is 10.
        verbose : bool, optional
            Verbose output. Default is True.
        random_state : int, optional
            Random state. Default is None.
        """
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_output = n_output
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize SimpleRNN
        self.rnn = SimpleRNN(n_nodes=n_nodes, n_features=n_features)
        
        # Output layer weights
        self.W_out = np.random.randn(n_nodes, n_output) * 0.01
        self.B_out = np.zeros(n_output)
        
        # Training history
        self.history = {'loss': []}
    
    def _activation_function(self, x):
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X):
        """
        Forward propagation.
        
        Parameters
        ----------
        X : ndarray, shape (batch_size, n_sequences, n_features)
            Input data
            
        Returns
        -------
        h : ndarray, shape (batch_size, n_nodes)
            Final hidden state
        y : ndarray, shape (batch_size, n_output)
            Output predictions
        """
        # Forward through RNN
        h = self.rnn.forward(X)
        
        # Output layer
        y = h @ self.W_out + self.B_out
        
        return h, y
    
    def backward(self, X, y_true, y_pred):
        """
        Backward propagation.
        
        Parameters
        ----------
        X : ndarray, shape (batch_size, n_sequences, n_features)
            Input data
        y_true : ndarray, shape (batch_size, n_output)
            True labels
        y_pred : ndarray, shape (batch_size, n_output)
            Predicted output
            
        Returns
        -------
        loss : float
            Computed loss
        """
        batch_size = X.shape[0]
        
        # Output layer gradients
        dy = y_pred - y_true  # (batch_size, n_output)
        dW_out = self.rnn.h_sequence[:, -1, :].T @ dy / batch_size  # (n_nodes, n_output)
        dB_out = np.mean(dy, axis=0)  # (n_output,)
        
        # Gradient to RNN output
        dh = dy @ self.W_out.T  # (batch_size, n_nodes)
        
        # Backward through RNN
        dW_x, dW_h, dB = self.rnn.backward(dh)
        
        # Update weights
        self.W_out -= self.lr * dW_out
        self.B_out -= self.lr * dB_out
        self.rnn.W_x -= self.lr * dW_x
        self.rnn.W_h -= self.lr * dW_h
        self.rnn.B -= self.lr * dB
        
        # Compute loss
        loss = np.mean((y_pred - y_true) ** 2)
        
        return loss
    
    def fit(self, X, y):
        """
        Train the classifier.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Training data
        y : ndarray, shape (n_samples, n_output)
            Training labels
            
        Returns
        -------
        self : ScratchSimpleRNNClassifier
            Returns self for method chaining
        """
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward
                _, y_pred = self.forward(X_batch)
                
                # Backward
                loss = self.backward(X_batch, y_batch, y_pred)
                epoch_loss += loss * (batch_end - i)
            
            avg_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_loss)
            
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Input data
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples, n_output)
            Predictions
        """
        _, y_pred = self.forward(X)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Input data
            
        Returns
        -------
        y_proba : ndarray, shape (n_samples, n_output)
            Predicted probabilities
        """
        _, y_pred = self.forward(X)
        
        # Apply sigmoid for binary classification or softmax for multi-class
        if self.n_output == 1:
            return 1 / (1 + np.exp(-y_pred))
        else:
            exp_y = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
            return exp_y / np.sum(exp_y, axis=1, keepdims=True)
