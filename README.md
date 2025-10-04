# machine_learning


## Playing Card Classifier

A PyTorch project for classifying playing cards (53 classes) using EfficientNet-B0 from timm with transfer learning.

### Features

    - Custom dataset class (PlayingCardDataset)
    - Train/validation/test splits with DataLoader
    - EfficientNet-B0 backbone + custom linear classifier
    - Training loop with Adam optimizer + CrossEntropyLoss
    - Loss curves visualization
    - Inference with probability bar chart

### setup

You can install the dependencies using pip:

  ```bash
pip install torch torchvision timm numpy pandas matplotlib pillow tqdm
 ```

### notes

    - Default image size: 128x128
    - Batch size: 32
    - Optimizer: Adam
    - Use GPU if available (cuda:0)

---



## Neural Network (NumPy & PyTorch)

This project implements and compares feedforward neural networks built **from scratch in NumPy** and **with PyTorch** (MLP + simple CNN). You’ll practice forward/backward propagation, SGD optimization, and evaluation on MNIST and Fashion-MNIST.

The objective is to first implement a neural network from scratch in NumPy and then reproduce it using PyTorch.

### Highlights

- **From-Scratch NumPy Net:** manual forward/backprop and SGD.
- **PyTorch Models:** MLP (`NeuralNetwork`) and CNN (`ConvolutionalNet`) with `nn.NLLLoss` and `SGD`.
- **Training Curves & Evaluation:** track/train/val loss, accuracy, and visualize samples.

### Data Preparation

- **Normalization:** custom min-max scaling.
- **Split:** 80% train / 20% validation.
- **Loaders:** PyTorch `DataLoader` for image datasets.

### Datasets

- MNIST: 60,000 training images + 10,000 test images of handwritten digits (28×28 pixels). - Fashion-MNIST: 60,000 grayscale images of clothing items in 10 categories (e.g., T-shirt, Sneaker, Bag).

### Neural Network Training

1.  Split data: 80% training / 20% validation.
2.  Build dataloaders for both sets.
3.  Use hyperparameters: learning_rate=0.005, num_epochs=5.
4.  Optimizer: SGD; Loss: Negative Log-Likelihood.
5.  Train on the training set and validate performance each epoch.
6.  Track and plot training/validation loss curves.
7.  Evaluate predictions and measure accuracy on the validation set.

---



## Logistic Regression

A hands-on notebook assignment to implement binary logistic regression from scratch (NumPy + Matplotlib) and apply it to two datasets:
student admission (linearly separable), and
microchip QA (non-linear; requires feature mapping + regularization).

### Section 1 - Logistic Regression

1. Load and visualize data (Logistic_Regression_data1.txt).
2. Implement:

- sigmoid(z)
- cost_function(theta, X, y)
- gradient_descent(theta, X, y)
- train_using_sgd(theta, X, y)

3. Plot decision boundary.
4. Implement predict and accuracy, report training accuracy.
5. Evaluate model with ROC curve + AUC.

### Section 2 — Regularized Logistic Regression (Microchips)

1. Load and visualize data (Logistic_Regression_data2.txt).
2. Implement map_feature(X1, X2, degree=6) for polynomial expansion.
3. Implement cost*function_gradient_descent_regularized(theta, X, y, lambda*).
4. Train using momentum (train_using_momentum).
5. Plot decision boundaries for λ=1 (regularized) vs. λ=0 (overfit).
6. Report and compare accuracy.

### Math Recap

Hypothesis:

$$
h_\theta(x) = \sigma(\theta^\top x)
$$

Sigmoid:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Unregularized Cost (Section 1):

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \Big[ -y^{(i)} \log \big(h_\theta(x^{(i)})\big) - \big(1 - y^{(i)}\big) \log \big(1 - h_\theta(x^{(i)})\big) \Big]
$$

Gradient (Unregularized):

$$
\nabla_\theta J(\theta) = \frac{1}{m} X^\top \big( h_\theta(X) - y \big)
$$

Regularized Cost (Section 2):

$$
J(\theta) = J_{\text{unreg}}(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

_(Note: do not regularize $\theta_0$)_

Regularized Gradient:
For $j = 0$:

$$
\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m \big(h_\theta(x^{(i)}) - y^{(i)}\big) x_0^{(i)}
$$

For $j \geq 1$:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \big(h_\theta(x^{(i)}) - y^{(i)}\big) x_j^{(i)} + \frac{\lambda}{m} \theta_j
$$

### Setup

 ```bash
 pip install numpy pandas matplotlib scikit-learn
  ```
