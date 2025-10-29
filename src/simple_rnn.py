import numpy as np


class SimpleRNN:
    """
    Simple Recurrent Neural Network implementation from scratch.
    
    This class follows the FC class structure as specified in the assignment.
    Implements RNN forward propagation and backpropagation using only NumPy.
    
    Parameters
    ----------
    n_nodes : int
        Number of RNN nodes
    n_features : int  
        Number of input features
    initializer : str, optional
        Weight initialization method. Default is 'random_normal'.
    """
    
    def __init__(self, n_nodes, n_features, initializer='random_normal'):
        self.n_nodes = n_nodes
        self.n_features = n_features
        
        # Initialize weights according to assignment formulas
        if initializer == 'random_normal':
            # W_x: Weight for input (n_features, n_nodes)
            self.W_x = np.random.randn(n_features, n_nodes) * 0.01
            # W_h: Weight for state (n_nodes, n_nodes) 
            self.W_h = np.random.randn(n_nodes, n_nodes) * 0.01
            # B: Bias term (n_nodes,)
            self.B = np.zeros(n_nodes)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
            
        # Store intermediate values for backpropagation
        self.x_t = None
        self.h_prev = None
        self.a_t = None  
        self.h_t = None
        
        # Store sequences for complete forward/backward passes
        self.x_sequence = None
        self.h_sequence = None
        self.a_sequence = None
    
    def forward(self, X):
        """
        Forward propagation of RNN.
        
        Parameters
        ----------
        X : ndarray, shape (batch_size, n_sequences, n_features)
            Input sequence data
            
        Returns
        -------
        h : ndarray, shape (batch_size, n_nodes) 
            Final hidden state
        """
        batch_size = X.shape[0]
        n_sequences = X.shape[1]
        
        # Initialize hidden state h_0 with zeros (as per assignment)
        h = np.zeros((batch_size, self.n_nodes))
        
        # Store sequences for backpropagation
        self.x_sequence = X
        self.h_sequence = np.zeros((batch_size, n_sequences, self.n_nodes))
        self.a_sequence = np.zeros((batch_size, n_sequences, self.n_nodes))
        
        # Process each time step following assignment formula:
        # a_t = x_t · W_x + h_{t-1} · W_h + B
        # h_t = tanh(a_t)
        for t in range(n_sequences):
            x_t = X[:, t, :]  # Input at time t (batch_size, n_features)
            a_t = x_t @ self.W_x + h @ self.W_h + self.B  # State before activation
            h = np.tanh(a_t)  # State and output at time t
            
            # Store for backpropagation
            self.h_sequence[:, t, :] = h
            self.a_sequence[:, t, :] = a_t
        
        return h
    
    def backward(self, d_h_t):
        """
        Backward propagation of RNN.
        
        Implements the exact formulas from the assignment:
        ∂h_t/∂a_t = ∂L/∂h_t × (1 − tanh²(a_t))
        ∂L/∂B = ∂h_t/∂a_t  
        ∂L/∂W_x = x_t^T · ∂h_t/∂a_t
        ∂L/∂W_h = h_{t-1}^T · ∂h_t/∂a_t
        
        ∂L/∂h_{t-1} = ∂h_t/∂a_t · W_h^T
        ∂L/∂x_t = ∂h_t/∂a_t · W_x^T
        
        Parameters
        ----------
        d_h_t : ndarray, shape (batch_size, n_nodes)
            Gradient of loss with respect to output h_t
            
        Returns
        -------
        dW_x : ndarray, shape (n_features, n_nodes)
            Gradient for input weights
        dW_h : ndarray, shape (n_nodes, n_nodes)  
            Gradient for hidden weights
        dB : ndarray, shape (n_nodes,)
            Gradient for bias
        """
        batch_size = self.x_sequence.shape[0]
        n_sequences = self.x_sequence.shape[1]
        
        # Initialize gradients
        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        dB = np.zeros_like(self.B)
        
        # Gradient from next time step (starts with output gradient)
        d_h_next = np.zeros((batch_size, self.n_nodes))
        
        # Backpropagate through time (reverse order)
        for t in range(n_sequences - 1, -1, -1):
            # Current values
            x_t = self.x_sequence[:, t, :]  # x_t: Input at time t
            a_t = self.a_sequence[:, t, :]  # a_t: Pre-activation
            
            if t == 0:
                h_prev = np.zeros((batch_size, self.n_nodes))  # h_{t-1} for t=0
            else:
                h_prev = self.h_sequence[:, t - 1, :]  # h_{t-1} for t>0
            
            # Assignment formula: ∂h_t/∂a_t = ∂L/∂h_t × (1 − tanh²(a_t))
            dL_dh_t = d_h_t + d_h_next  # ∂L/∂h_t (sum of output and previous state error)
            dh_t_da_t = dL_dh_t * (1 - np.tanh(a_t) ** 2)  # ∂h_t/∂a_t
            
            # Assignment formulas for gradients:
            # ∂L/∂B = ∂h_t/∂a_t
            dB += np.sum(dh_t_da_t, axis=0)  # (n_nodes,)
            
            # ∂L/∂W_x = x_t^T · ∂h_t/∂a_t  
            dW_x += x_t.T @ dh_t_da_t  # (n_features, n_nodes)
            
            # ∂L/∂W_h = h_{t-1}^T · ∂h_t/∂a_t
            dW_h += h_prev.T @ dh_t_da_t  # (n_nodes, n_nodes)
            
            # Assignment formulas for errors to previous time/layers:
            # ∂L/∂h_{t-1} = ∂h_t/∂a_t · W_h^T
            d_h_next = dh_t_da_t @ self.W_h.T  # (batch_size, n_nodes)
            
            # ∂L/∂x_t = ∂h_t/∂a_t · W_x^T (if needed for previous layers)
            # d_x_t = dh_t_da_t @ self.W_x.T
        
        # Average gradients over batch
        dW_x /= batch_size
        dW_h /= batch_size  
        dB /= batch_size
        
        return dW_x, dW_h, dB
