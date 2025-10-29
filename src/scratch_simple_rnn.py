import numpy as np


class ScratchSimpleRNNClassifier:
    """
    Simple RNN classifier implemented from scratch.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the RNN layer
    n_features : int
        Number of input features
    n_output : int, optional
        Number of output classes. Default is 1.
    activation : str, optional
        Activation function to use. 'tanh' or 'relu'. Default is 'tanh'.
    optimizer : str, optional
        Optimizer to use. 'sgd'. Default is 'sgd'.
    lr : float, optional
        Learning rate. Default is 0.01.
    batch_size : int, optional
        Batch size for training. Default is 32.
    epochs : int, optional
        Number of training epochs. Default is 10.
    verbose : bool, optional
        If True, print training progress. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    """
    
    def __init__(self, n_nodes, n_features, n_output=1, activation='tanh', 
                 optimizer='sgd', lr=0.01, batch_size=32, epochs=10, 
                 verbose=True, random_state=None):
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_output = n_output
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        
        # Initialize weights and biases
        if random_state is not None:
            np.random.seed(random_state)
            
        # Input weights: (n_features, n_nodes)
        self.W_x = np.random.normal(0, 0.01, (n_features, n_nodes))
        # Hidden state weights: (n_nodes, n_nodes)
        self.W_h = np.random.normal(0, 0.01, (n_nodes, n_nodes))
        # Bias: (n_nodes,)
        self.B = np.zeros(n_nodes)
        
        # Output layer weights and bias
        self.W_out = np.random.normal(0, 0.01, (n_nodes, n_output))
        self.B_out = np.zeros(n_output)
        
        # Store intermediate values for backpropagation
        self.history = {'loss': [], 'accuracy': []}
        
    def _activation_function(self, x):
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def _activation_derivative(self, x):
        """Derivative of activation function."""
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def forward(self, X):
        """
        Forward propagation of the RNN.
        
        Parameters
        ----------
        X : ndarray, shape (batch_size, n_sequences, n_features)
            Input data
            
        Returns
        -------
        h : ndarray, shape (batch_size, n_nodes)
            Final hidden state
        h_sequence : ndarray, shape (batch_size, n_sequences, n_nodes)
            All hidden states
        y : ndarray, shape (batch_size, n_output)
            Output predictions
        """
        batch_size = X.shape[0]
        n_sequences = X.shape[1]
        
        # Initialize hidden state with zeros
        h = np.zeros((batch_size, self.n_nodes))
        
        # Store all hidden states for backpropagation
        h_sequence = np.zeros((batch_size, n_sequences, self.n_nodes))
        
        # Forward propagation through sequences
        for t in range(n_sequences):
            x_t = X[:, t, :]  # (batch_size, n_features)
            a = x_t @ self.W_x + h @ self.W_h + self.B  # (batch_size, n_nodes)
            h = self._activation_function(a)  # (batch_size, n_nodes)
            h_sequence[:, t, :] = h
        
        # Output layer
        y = h @ self.W_out + self.B_out  # (batch_size, n_output)
        
        # Store for backpropagation
        self.X = X
        self.h_sequence = h_sequence
        self.h = h
        self.y = y
        
        return h, h_sequence, y
    
    def backward(self, X, y_true, y_pred):
        """
        Backward propagation of the RNN.
        
        Parameters
        ----------
        X : ndarray, shape (batch_size, n_sequences, n_features)
            Input data
        y_true : ndarray, shape (batch_size, n_output)
            True labels
        y_pred : ndarray, shape (batch_size, n_output)
            Predicted output
        """
        batch_size = X.shape[0]
        n_sequences = X.shape[1]
        
        # Output layer gradients
        dy = y_pred - y_true  # (batch_size, n_output)
        dW_out = self.h.T @ dy / batch_size  # (n_nodes, n_output)
        dB_out = np.mean(dy, axis=0)  # (n_output,)
        
        # Gradient flowing back from output
        dh = dy @ self.W_out.T  # (batch_size, n_nodes)
        
        # Initialize gradients
        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        dB = np.zeros_like(self.B)
        
        # Backpropagate through time
        dh_next = np.zeros((batch_size, self.n_nodes))
        
        for t in range(n_sequences - 1, -1, -1):
            if t == 0:
                h_prev = np.zeros((batch_size, self.n_nodes))
            else:
                h_prev = self.h_sequence[:, t - 1, :]
            
            x_t = X[:, t, :]
            h_t = self.h_sequence[:, t, :]
            
            # Compute a_t (pre-activation)
            a_t = x_t @ self.W_x + h_prev @ self.W_h + self.B
            
            # Derivative of activation function
            da = self._activation_derivative(a_t) * (dh + dh_next)
            
            # Gradients
            dW_x += x_t.T @ da / batch_size
            dW_h += h_prev.T @ da / batch_size
            dB += np.mean(da, axis=0)
            
            # Gradient for previous time step
            dh_next = da @ self.W_h.T
        
        # Store gradients for debugging
        self.W_x_grad = dW_x
        self.W_h_grad = dW_h
        self.B_grad = dB
        self.W_out_grad = dW_out
        self.B_out_grad = dB_out
        
        # Update weights using gradient descent
        self.W_out -= self.lr * dW_out
        self.B_out -= self.lr * dB_out
        self.W_x -= self.lr * dW_x
        self.W_h -= self.lr * dW_h
        self.B -= self.lr * dB
        
        # Compute loss for monitoring
        loss = np.mean((y_pred - y_true) ** 2)
        
        return loss
    
    def fit(self, X, y):
        """
        Train the RNN classifier.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Training data
        y : ndarray, shape (n_samples, n_output)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_end = i + self.batch_size
                if batch_end > n_samples:
                    batch_end = n_samples
                
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward pass
                _, _, y_pred = self.forward(X_batch)
                
                # Backward pass
                loss = self.backward(X_batch, y_batch, y_pred)
                epoch_loss += loss * (batch_end - i)
            
            avg_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_loss)
            
            if self.verbose and (epoch % 1 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Predict using the trained RNN.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Input data
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples, n_output)
            Predicted values
        """
        _, _, y_pred = self.forward(X)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained RNN.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_sequences, n_features)
            Input data
            
        Returns
        -------
        y_proba : ndarray, shape (n_samples, n_output)
            Predicted probabilities
        """
        _, _, y_pred = self.forward(X)
        # Apply sigmoid for binary classification or softmax for multiclass
        if self.n_output == 1:
            return 1 / (1 + np.exp(-y_pred))
        else:
            exp_y = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
            return exp_y / np.sum(exp_y, axis=1, keepdims=True)
