#  Python Machine Learning Roadmap

This repository tracks my journey through the fundamentals of Artificial Intelligence—from the core mathematics of neural networks to real-world data science applications using Scikit-learn and Pandas.

##  Project Milestones

The project is structured into three progressive levels of complexity:

### 1. Simple Neural Network (`neural_network_v1.py`)
- **Type**: Single-layer Perceptron built from scratch with NumPy.
- **Task**: Solving the **AND** logic gate.
- **Key Concepts**: Weights, bias, Sigmoid activation function, and basic feedforward training.

### 2. Multi-Layer Network - XOR Solver (`neural_network_v2.py`)
- **Type**: Multi-layer Perceptron (MLP) with a Hidden Layer.
- **Task**: Solving the **XOR** (Exclusive OR) problem, which is not linearly separable.
- **Key Concepts**: Backpropagation algorithm and how hidden layers allow networks to learn non-linear relationships.

### 3. Titanic Survival Predictor (`titanic_realistic.py`)
- **Type**: Supervised Learning using professional libraries.
- **Task**: Predicting passenger survival chances based on historical data (age, sex, ticket class).
- **Model**: Random Forest Classifier.
- **Performance**: Achieves a realistic accuracy of **~78-82%**.
- **Key Concepts**: Data preprocessing with Pandas, handling missing values, train/test splitting, and ensemble modeling.

##  Requirements & Installation

To run these scripts, you need Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn
