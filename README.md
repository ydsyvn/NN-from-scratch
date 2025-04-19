# Simple Feedforward Neural Network in NumPy

This notebook provides a basic implementation of a two-layer feedforward neural network built from scratch using only NumPy. It's designed for educational purposes and demonstrating the fundamentals of neural network mechanics for multi-class classification tasks (like MNIST).

## Features

* **Two-Layer Architecture:** Input layer, one hidden layer with ReLU activation, and an output layer with Softmax activation.
* **NumPy Implementation:** Built entirely using NumPy for core computations.
* **Xavier/Glorot Initialization:** Uses Xavier/Glorot method for initializing weights.
* **Forward Propagation:** Calculates network output.
* **Backward Propagation:** Implements backpropagation with categorical cross-entropy loss for gradient calculation.
* **Mini-Batch Gradient Descent:** Includes a training loop that uses mini-batches.
* **Metrics Tracking:** Records training and test loss/accuracy per epoch.
* **Basic Visualization:** Helper functions to plot training history and visualize predictions (requires Matplotlib).

## Requirements

* Python 3.x
* NumPy
* Matplotlib (for visualization functions)

## Usage

1.  **Instantiate the Network:**
    ```python
    import numpy as np
    # Assuming matplotlib.pyplot is imported as plt for visualizations

    # Load and preprocess your data (e.g., MNIST)
    # Ensure X_train, y_train (one-hot), X_test, y_test (one-hot) are prepared

    input_size = 784  # Example: MNIST image dimensions (28*28)
    hidden_size = 128 # Example: Number of neurons in hidden layer
    output_size = 10  # Example: Number of classes (digits 0-9)
    learning_rate = 0.01

    nn = DeepNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    ```

2.  **Train the Network:**
    ```python
    epochs = 15
    batch_size = 64

    nn.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
    ```

3.  **Evaluate and Predict:**
    ```python
    # Plot training history
    nn.plot_training_history()

    # Make predictions on new data
    # predictions = nn.predict(X_new)

    # Visualize some test predictions
    # nn.visualize_predictions(X_test, np.argmax(y_test, axis=1), samples=5) # Pass original labels if needed for title
    ```

---

*Note: This implementation is intended for learning. For more complex tasks or production use, consider using established deep learning frameworks like TensorFlow or PyTorch.*
