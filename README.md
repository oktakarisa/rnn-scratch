# RNN from Scratch Implementation

This project implements a Simple Recurrent Neural Network (RNN) from scratch using only NumPy, as required for the RNN assignment.

## Assignment Overview

This project implements a Recurrent Neural Network with the following requirements:

1. **Create `ScratchSimpleRNNClassifier` class** with forward propagation
2. **Test forward propagation** with provided small sequence example 
3. **(Advanced) Implement backpropagation** with proper gradient calculation

## Project Structure

```
rnn-scratch/
│
├── data/                # (optional) for toy datasets if used
├── plots/               # visualizations of outputs or losses
├── reports/             # experiment notes or generated reports
├── src/                 # implementation scripts
│   ├── __init__.py
│   └── scratch_simple_rnn.py   # class ScratchSimpleRNNClassifier
│
├── main.py              # main executable script
├── rnn-scratch.ipynb    # notebook for same assignment
├── requirements.txt     # numpy, matplotlib (optional)
├── README.md            # documentation and usage
└── .gitignore           # ignore unnecessary files (optional)
```

## RNN Implementation Details

### Forward Propagation Formula

The RNN forward propagation follows these mathematical formulas:

$$a_t = x_t \cdot W_x + h_{t-1} \cdot W_h + B$$
$$h_t = \tanh(a_t)$$

Where:
- $a_t$: State before activation at time t (batch_size, n_nodes)
- $h_t$: State/output at time t (batch_size, n_nodes)
- $x_t$: Input at time t (batch_size, n_features)
- $W_x$: Input weights (n_features, n_nodes)
- $W_h$: Hidden state weights (n_nodes, n_nodes)
- $B$: Bias term (n_nodes,)

### Backpropagation Implementation

The backpropagation implementation includes:

$$\frac{\partial L}{\partial B} = \frac{\partial h_t}{\partial a_t}$$
$$\frac{\partial L}{\partial W_x} = x_t^T \cdot \frac{\partial h_t}{\partial a_t}$$
$$\frac{\partial L}{\partial W_h} = h_{t-1}^T \cdot \frac{\partial h_t}{\partial a_t}$$

Weight updates use gradient descent:

$$W_x' = W_x - \alpha \frac{\partial L}{\partial W_x}$$
$$W_h' = W_h - \alpha \frac{\partial L}{\partial W_h}$$
$$B' = B - \alpha \frac{\partial L}{\partial B}$$

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Assignment Requirements Verification

**For exact assignment compliance**, run:

```bash
python test_assignment_requirements.py
```

This will:
1. Test forward propagation with the exact small sequence example from the assignment
2. Verify backpropagation formulas match the assignment exactly
3. Confirm SimpleRNN class follows FC-class structure
4. Check Python 3.7.x compatibility

### Using the Python Script

Run the main executable to test all functionality:

```bash
python main.py
```

This will:
1. Test forward propagation with the assignment's small sequence example
2. Test training functionality on synthetic data
3. Test backpropagation implementation

### Using the Jupyter Notebook

Open the notebook to explore the implementation interactively:

```bash
jupyter notebook rnn-scratch.ipynb
```

The notebook includes:
- Detailed implementation walkthrough
- Step-by-step forward propagation test
- Training demonstration with visualization
- Backpropagation verification

### Using the RNN Class Directly

```python
import numpy as np
from src.scratch_simple_rnn import ScratchSimpleRNNClassifier

# Create RNN instance
rnn = ScratchSimpleRNNClassifier(
    n_nodes=10,              # Number of RNN nodes
    n_features=3,            # Input feature dimension
    n_output=1,              # Output dimension
    activation='tanh',       # Activation function
    lr=0.01,                 # Learning rate
    batch_size=32,           # Batch size
    epochs=20,               # Training epochs
    verbose=True,            # Print progress
    random_state=42          # Reproducibility
)

# Training data
X = np.random.randn(100, 5, 3)  # (n_samples, n_sequences, n_features)
y = np.random.randn(100, 1)     # (n_samples, n_output)

# Train the model
rnn.fit(X, y)

# Make predictions
predictions = rnn.predict(X)
probabilities = rnn.predict_proba(X)
```

## Test Results

### Forward Propagation Test

Using the assignment's exact test values:

```
Expected output: [0.79494228, 0.81839002, 0.83939649, 0.85584174]
Computed output: [0.79494228, 0.81839002, 0.83939649, 0.85584174]
✅ PASSED: Results match within tolerance
```

### Training Test

The implementation successfully:
- Trains on synthetic sequence classification data
- Decreases loss consistently over epochs  
- Achieves reasonable training accuracy
- Generates plots for training loss and predictions

### Backpropagation Test

The backpropagation implementation:
- Computes gradients for all weights and biases
- Validates gradient tensor shapes
- Performs proper weight updates

## Key Features

- **Pure NumPy Implementation**: No deep learning frameworks used
- **Complete RNN Implementation**: Forward and backward propagation
- **Mini-batch Training**: Support for batch processing
- **Flexible Architecture**: Configurable nodes, features, and output dimensions
- **Multiple Activation Functions**: Support for tanh and ReLU
- **Probability Prediction**: Sigmoid/softmax for classification
- **Comprehensive Testing**: Validates all components against assignment requirements

## Architecture

The `ScratchSimpleRNNClassifier` class includes:

1. **Initialization**: Weight and bias initialization
2. **Forward Propagation**: Sequential processing with hidden state
3. **Backpropagation**: BPTT (Backpropagation Through Time)
4. **Training**: Mini-batch gradient descent
5. **Prediction**: Direct inference and probability estimation

## Limitations

- Simple architecture without advanced regularization
- Limited to basic activation functions
- No built-in cross-validation or hyperparameter tuning
- Minimal optimization compared to framework implementations

## Future Enhancements

- Add more activation functions (sigmoid, leaky ReLU, etc.)
- Implement L2 regularization
- Add dropout for regularization
- Include more optimizers (Adam, RMSprop, etc.)
- Add support for bidirectional RNNs
- Implement LSTM/GRU variants

## Assignment Compliance

This implementation fully satisfies the assignment requirements:

✅ **Problem 1**: Simple forward propagation implementation of RNN  
✅ **Problem 2**: Experiment of forward propagation with small sequence  
✅ **Problem 3**: Implementation of backpropagation (advanced assignment)  

All formulas follow the exact mathematical specifications in the assignment, and the test with the provided small sequence produces the expected output.
