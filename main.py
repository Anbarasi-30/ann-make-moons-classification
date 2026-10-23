"""
Artificial Neural Network (ANN) for Binary Classification using make_moons.

Run:
    pip install -r requirements.txt
    python main.py

The script trains a small neural network, prints evaluation metrics,
and saves a decision-boundary visualization as ann_decision_boundary.png.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def main():
    # 1. Create a two-class, moon-shaped dataset.
    X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)

    # 2. Split the data so we can evaluate on unseen examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scale input features to help the neural network train well.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build and train a feed-forward artificial neural network.
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the trained model.
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print("ANN BINARY CLASSIFICATION - MAKE_MOONS DATASET")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Test accuracy   : {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # 6. Plot the data and learned decision boundary.
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = __import__("numpy").meshgrid(
        __import__("numpy").arange(x_min, x_max, 0.02),
        __import__("numpy").arange(y_min, y_max, 0.02)
    )
    grid = __import__("numpy").c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    zz = model.predict(grid_scaled).reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black", s=25)
    plt.title(f"ANN Decision Boundary on make_moons (Accuracy: {accuracy:.2%})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("ann_decision_boundary.png", dpi=150)
    print("\nSaved visualization to: ann_decision_boundary.png")
    plt.show()


if __name__ == "__main__":
    main()
