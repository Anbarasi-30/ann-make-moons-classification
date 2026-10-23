# ANN Binary Classification using `make_moons`

## Project overview
This project demonstrates a small Artificial Neural Network (ANN) that learns to classify two classes in the Scikit-learn `make_moons` dataset. The two classes form curved, moon-like shapes, so the model must learn a non-linear decision boundary.

## Objectives
- Generate a binary classification dataset using Scikit-learn.
- Split the data into training and testing sets.
- Standardize the input features.
- Train a feed-forward neural network.
- Evaluate the model with accuracy, a classification report, and a confusion matrix.
- Visualize the learned decision boundary.

## Technologies
- Python
- Scikit-learn
- Matplotlib

## Project files
- `main.py` — data generation, ANN training, evaluation, and visualization.
- `requirements.txt` — required Python packages.
- `ann_decision_boundary.png` — generated automatically when the program runs.

## How to run

### 1. Install Python
Use Python 3.10 or newer if possible.

### 2. Install dependencies
Open a terminal in this project folder and run:

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

The program prints the test accuracy, classification report, and confusion matrix. It also saves `ann_decision_boundary.png` in the project folder and displays the plot.

## Model details
- Dataset: `make_moons` with 1,000 samples and noise 0.20.
- Train/test split: 80% training and 20% testing.
- Preprocessing: `StandardScaler`.
- Neural network: two hidden layers with 16 and 8 neurons.
- Hidden activation: ReLU.
- Optimizer: Adam.
- Evaluation: accuracy, precision, recall, F1-score, and confusion matrix.

## Expected result
The model should learn a curved boundary that separates the two moon-shaped classes. Exact accuracy can vary slightly with library versions and training behavior; use the value printed by your own run as the actual result.

## Learning outcome
This project illustrates how an ANN can learn a non-linear relationship from data and how a held-out test set can be used to evaluate classification performance.

## Author
ANBARASI S
