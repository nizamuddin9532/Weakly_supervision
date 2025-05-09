import numpy as np
from scipy.special import expit as sigmoid

class WeakSupervisionEM:
    def __init__(self, n_epochs=100, tol=1e-4):
        self.n_epochs = n_epochs
        self.tol = tol
        
    def fit(self, L):
        self.L = L
        self.n_samples, self.n_sources = L.shape
        self.theta = np.random.rand(self.n_sources)  # Source accuracies
        self.phi = np.random.rand(self.n_samples)    # True label probabilities
        
        prev_ll = -np.inf
        for epoch in range(self.n_epochs):
            self._e_step()
            self._m_step()
            
            current_ll = self._log_likelihood()
            if np.abs(current_ll - prev_ll) < self.tol:
                break
            prev_ll = current_ll
            
        return self
    
    def _e_step(self):
        for i in range(self.n_samples):
            numerator = 1.0
            denominator = 1.0
            
            for j in range(self.n_sources):
                if self.L[i,j] != 0:
                    prob_if_positive = sigmoid(self.theta[j]) if self.L[i,j] > 0 else 1 - sigmoid(self.theta[j])
                    prob_if_negative = sigmoid(self.theta[j]) if self.L[i,j] < 0 else 1 - sigmoid(self.theta[j])
                    
                    numerator *= prob_if_positive
                    denominator *= prob_if_negative
            
            self.phi[i] = numerator / (numerator + denominator)
    
    def _m_step(self):
        for j in range(self.n_sources):
            numerator = 0.0
            denominator = 0.0
            
            for i in range(self.n_samples):
                if self.L[i,j] != 0:
                    if self.L[i,j] > 0:
                        numerator += self.phi[i]
                        denominator += 1.0
                    else:
                        numerator += (1 - self.phi[i])
                        denominator += 1.0
            
            if denominator > 0:
                accuracy = numerator / denominator
                self.theta[j] = np.log(accuracy / (1 - accuracy))
    
    def _log_likelihood(self):
        ll = 0.0
        for i in range(self.n_samples):
            term = 0.0
            for j in range(self.n_sources):
                if self.L[i,j] != 0:
                    prob_if_positive = sigmoid(self.theta[j]) if self.L[i,j] > 0 else 1 - sigmoid(self.theta[j])
                    prob_if_negative = sigmoid(self.theta[j]) if self.L[i,j] < 0 else 1 - sigmoid(self.theta[j])
                    term += np.log(self.phi[i] * prob_if_positive + (1 - self.phi[i]) * prob_if_negative)
            ll += term
        return ll
    
    def predict_proba(self):
        return self.phi
    
    def predict(self, threshold=0.5):
        return (self.phi >= threshold).astype(int)