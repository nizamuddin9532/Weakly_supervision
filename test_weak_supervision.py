import numpy as np
from weak_supervision import WeakSupervisionEM
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=1000, n_sources=5):
    """Generate synthetic weak supervision data"""
    true_labels = np.random.randint(0, 2, size=n_samples)
    L = np.zeros((n_samples, n_sources))
    accuracies = [0.7, 0.8, 0.6, 0.9, 0.75]  # True accuracies
    
    for j in range(n_sources):
        for i in range(n_samples):
            if np.random.rand() < 0.5:  # 50% chance of labeling
                if true_labels[i] == 1:
                    L[i,j] = 1 if np.random.rand() < accuracies[j] else -1
                else:
                    L[i,j] = -1 if np.random.rand() < accuracies[j] else 1
    return L, true_labels

def plot_results(true_labels, predicted_probs, model):
    """Visualize the EM algorithm results"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Probability distributions
    plt.subplot(1, 2, 1)
    plt.hist(predicted_probs[true_labels == 1], bins=20, alpha=0.7, label='True Positive', color='blue')
    plt.hist(predicted_probs[true_labels == 0], bins=20, alpha=0.7, label='True Negative', color='red')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution by True Label")
    plt.legend()
    
    # Plot 2: Source accuracies
    plt.subplot(1, 2, 2)
    true_accuracies = [0.7, 0.8, 0.6, 0.9, 0.75]  # For comparison
    bar_width = 0.35
    x = np.arange(len(model.theta))
    
    plt.bar(x - bar_width/2, true_accuracies, bar_width, label='True Accuracy', alpha=0.7)
    plt.bar(x + bar_width/2, sigmoid(model.theta), bar_width, label='Estimated Accuracy', alpha=0.7)
    plt.xlabel("Source Index")
    plt.ylabel("Accuracy")
    plt.title("Source Accuracy Comparison")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate synthetic data
    L, true_labels = generate_synthetic_data()
    
    # Run EM algorithm
    model = WeakSupervisionEM(n_epochs=50)
    model.fit(L)
    
    # Get predictions
    predicted_probs = model.predict_proba()
    predicted_labels = model.predict()
    
    # Evaluate
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy:.3f}")
    print("Estimated source accuracies:", [f"{x:.3f}" for x in sigmoid(model.theta)])
    
    # Visualize results
    plot_results(true_labels, predicted_probs, model)

if __name__ == "__main__":
    main()