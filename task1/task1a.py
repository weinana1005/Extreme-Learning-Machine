import math
import torch
import torch.nn as nn
import torch.optim as optim


def compute_feature_groups(X, max_M):
    """
    Compute polynomial feature groups for each degree from 0 to max_M.
    Returns a list where each element is a tensor of shape (N, p_m) corresponding to features of degree m.
    """
    N, D = X.shape
    groups = []
    # Degree 0: constant term
    groups.append(torch.ones((N, 1), dtype=X.dtype, device=X.device))

    if max_M >= 1:
        # Degree 1: linear terms xi
        groups.append(X)

    if max_M >= 2:
        # Degree 2: squares xi^2 + pairwise interactions xixj
        squares = X ** 2
        cross_terms = []
        for i in range(D):
            for j in range(i + 1, D):
                cross_terms.append((X[:, i] * X[:, j]).unsqueeze(1))
        if cross_terms:
            interactions = torch.cat(cross_terms, dim=1)
            group2 = torch.cat([squares, interactions], dim=1)
        else:
            group2 = squares
        groups.append(group2)

    if max_M >= 3:
        # Degree 3: cubes xi^3 + mixed terms xi^2xj + triple products xixjxk
        cubes = X ** 3
        mix1 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix1.append((X[:, i] ** 2 * X[:, j]).unsqueeze(1))
        mix2 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix2.append((X[:, i] * X[:, j] ** 2).unsqueeze(1))
        triple = []
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    triple.append((X[:, i] * X[:, j] * X[:, k]).unsqueeze(1))
        group3 = cubes
        if mix1:
            group3 = torch.cat([group3, torch.cat(mix1, dim=1)], dim=1)
        if mix2:
            group3 = torch.cat([group3, torch.cat(mix2, dim=1)], dim=1)
        if triple:
            group3 = torch.cat([group3, torch.cat(triple, dim=1)], dim=1)
        groups.append(group3)

    return groups

# Instead of concatenating everything into one big tensor, we keep degree-specific features separate.
# This allows models (like the LearnableLogisticModel) to assign different weights or gating parameters to each polynomial degree,
# effectively learning how much each degree contributes.

class LearnableLogisticModel(nn.Module):                           # Inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
                                                                   # Now we can use PyTorch's built in function
    """
    Logistic regression model with learnable weighting over polynomial feature groups.
    For each degree m (from 0 to max_M), polynomial features are computed and weighted by a learnable
    parameter. The final output is the weighted sum of contributions from each group, passed through a sigmoid.
    """

    def __init__(self, D, max_M):
        """
        Args:
            D (int): Dimensionality of input.
            max_M (int): Maximum polynomial degree to consider.
        """
        super(LearnableLogisticModel, self).__init__()              # Calls the constructor of the parent class (nn.Module) to properly initialize the module.
        self.D = D
        self.max_M = max_M

        # Determine the number of features for each degree.
        self.group_feature_sizes = []

        # Degree 0: constant term (1 feature)
        self.group_feature_sizes.append(1)

        # Degree 1: linear (D features)
        if max_M >= 1:
            self.group_feature_sizes.append(D)

        # Degree 2: squares (D) + cross terms (D choose 2)
        if max_M >= 2:
            cross_terms = (D * (D - 1)) // 2
            self.group_feature_sizes.append(D + cross_terms)

        # Degree 3: cubes (D) + mix terms (2 * (D choose 2)) + triple cross terms (D choose 3)
        if max_M >= 3:
            mix_terms = 2 * ((D * (D - 1)) // 2)
            triple = (D * (D - 1) * (D - 2)) // 6
            self.group_feature_sizes.append(D + mix_terms + triple)

        # Create a weight parameter for each degree group.
        self.weights = nn.ParameterList()                            # one ParameterList entry per degree
        for size in self.group_feature_sizes:
            # Initialize each weight vector with zeros.
            w = nn.Parameter(torch.zeros(size, 1))
            self.weights.append(w)

        # Each group has a separate weight matrix.
        # size is the number of features in that group, and we store a (size, 1) weight vector for them.
        # We initialize with zeros (though you could choose random initialization).
        # By putting them in a nn.ParameterList, PyTorch knows these are trainable parameters.

        # Gating parameters (one per degree group), which are unconstrained.
        self.alpha = nn.Parameter(torch.zeros(len(self.group_feature_sizes)))  # Shape: 1D tensor (max_M+1,)

# The LearnableLogisticModel class:
# Tracks how many features each polynomial degree group has.
# Allocates a trainable weight vector for each group.
# Introduces gating parameters (alpha) to weight each group’s contribution in a learnable way.
# By doing so, it effectively learns an “optimal” polynomial order (or mixture of orders) for logistic regression.

    def forward(self, X):
        """
        Forward pass.
        Args:
            X (torch.Tensor): Input tensor of shape (N, D).
        Returns:
            torch.Tensor: Predicted probabilities of shape (N,).
        """
        # This line splits the input X (shape (N, D)) into several groups of polynomial features based on degrees 0 to max_M.
        # Each group is a tensor of shape (N, p_m) containing features for a specific degree (e.g., constant, linear, quadratic, cubic).
        groups = compute_feature_groups(X, self.max_M)

        contributions = []

        # For each degree, compute the dot product of the features with the corresponding weight vector.
        for m, feat in enumerate(groups):
            # feat: (N, p_m); self.weights[m]: (p_m, 1)
            contrib = feat.matmul(self.weights[m])
            contributions.append(contrib)

        # Concatenate contributions: shape (N, number_of_groups)
        contributions = torch.cat(contributions, dim=1)  # shape (N, G)

        # Compute gating weights via softmax.
        # Applying softmax converts these values into a probability distribution over the groups,
        # meaning the weights sum to 1 and determine each group’s relative importance.
        gating = torch.softmax(self.alpha, dim=0)  # (G,)

        # Compute the weighted sum over groups for each sample.
        # matrix(N,G) * matrix(G,) = matrix(N,)
        f = (contributions * gating).sum(dim=1)    # (N,)

        # sigmoid squashes each value in f to lie between 0 and 1.
        # yields the predicted probabilities for each input sample.
        return torch.sigmoid(f)

    def effective_degree(self):
        """
        Computes an effective polynomial degree as the weighted average of the degrees.
        Returns:
            tuple: (effective_degree (float), gating_weights (list of floats))
        """
        # Gating Weights Calculations:
        # Applies the softmax function to self.alpha along dimension 0. This converts the raw gating parameters into normalized weights that sum to 1. Each weight corresponds to a polynomial degree group.
        # Detaches the tensor from the current computation graph, ensuring that subsequent operations do not track gradients.
        # Moves the tensor to the CPU (if it was on a GPU).
        # Converts the PyTorch tensor to a NumPy. e.g. [0.1, 0.4, 0.3, 0.2] if 4 groups
        gating = torch.softmax(self.alpha, dim=0).detach().cpu().numpy()

        # This line creates a list of integers representing the degrees corresponding to each group.
        # e.g. 4 groups become [0, 1, 2, 3]
        degrees = [m for m in range(len(gating))]

        # Sum the products (g * m)
        # zip will make them pair up.
        # e.g.gating = [0.1, 0.4, 0.3, 0.2] degrees = [0, 1, 2, 3].  zip(gating, degrees) = (0.1,0), (0.4,1)......
        effective = sum(g * m for g, m in zip(gating, degrees))
        return effective, gating

# The forward method defines how the model processes an input tensor X and produces the predicted probabilities.


# ---------------------------------------------
# Helper Functions (from task1)
# ---------------------------------------------
def accuracy(y_pred, t):
    """
    Compute classification accuracy given predicted probabilities and true binary targets.
    """
    preds = (y_pred >= 0.5).float()
    return (preds == t).float().mean().item()


def compute_features(X, M):
    N, D = X.shape
    features = []
    features.append(torch.ones((N, 1), dtype=X.dtype, device=X.device))
    if M >= 1:
        features.append(X)
    if M >= 2:
        squares = X ** 2
        cross_terms = []
        for i in range(D):
            for j in range(i + 1, D):
                cross_terms.append((X[:, i] * X[:, j]).unsqueeze(1))
        if cross_terms:
            features.append(torch.cat([squares, torch.cat(cross_terms, dim=1)], dim=1))
    if M >= 3:
        cubes = X ** 3
        mix1 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix1.append((X[:, i] ** 2 * X[:, j]).unsqueeze(1))
        mix2 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix2.append((X[:, i] * X[:, j] ** 2).unsqueeze(1))
        triple = []
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    triple.append((X[:, i] * X[:, j] * X[:, k]).unsqueeze(1))
        feat3 = cubes
        if mix1:
            feat3 = torch.cat([feat3, torch.cat(mix1, dim=1)], dim=1)
        if mix2:
            feat3 = torch.cat([feat3, torch.cat(mix2, dim=1)], dim=1)
        if triple:
            feat3 = torch.cat([feat3, torch.cat(triple, dim=1)], dim=1)
        features.append(feat3)
    return torch.cat(features, dim=1)


def generate_underlying_weights(M, D):
    """
    Generate an underlying weight vector for data generation (using fixed M, here M=2).
    """
    dummy = torch.zeros(1, D)
    p = compute_features(dummy, M).shape[1]
    w_true = torch.zeros(p)
    for i in range(p):
        k = p - i
        w_true[i] = ((-1) ** k) * math.sqrt(k) / p
    return w_true


def generate_dataset(N, D, w_true, noise_std=1.0):
    """
    Generate a dataset with N examples in D dimensions using the underlying true model (with M=2).
    """
    X = (10 * torch.rand(N, D)) - 5
    y_true = torch.sigmoid(compute_features(X, 2).matmul(w_true))
    noise = torch.normal(mean=0.0, std=noise_std, size=y_true.shape)
    y_noisy = y_true + noise
    t = (y_noisy >= 0.5).float()
    return X, t

#---------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Settings for data generation and training.
    D = 5
    N_train = 200
    N_test = 100
    M_gen = 2  # Use fixed M=2 for data generation.

    # Generate underlying weights for data generation.
    w_true = generate_underlying_weights(M_gen, D)
    print("Underlying weights for data generation (M=2):")
    print(w_true)

    # Generate training and test datasets.
    X_train, t_train = generate_dataset(N_train, D, w_true)
    X_test, t_test = generate_dataset(N_test, D, w_true)

    # Create the learnable logistic model with max_M=3 (degrees 0,1,2,3).
    max_M = 3
    model = LearnableLogisticModel(D, max_M)

    # Sets up the binary cross-entropy loss function. This loss is appropriate for binary classification tasks.
    loss_fn = nn.BCELoss()
    # Creates an SGD optimizer that will update all trainable parameters of the model (weight matrices & gating parameters) with a learning rate of 0.01.
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training Hyperparameters
    epochs = 200
    batch_size = 32
    N = X_train.shape[0]    # Number of training samples
    print_interval = max(1, epochs // 10)

    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(N)                                    # randomly shuffles the indices of the training data for each epoch.
        epoch_loss = 0.0
        count = 0
        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]                        # Indices will be from i to i + batch size
            # Extract Mini-Batch Data
            X_batch = X_train[indices]                                     # Extracts a mini-batch of inputs (X_batch) using the shuffled indices.
            t_batch = t_train[indices]                                     # Extracts targets (t_batch) using the shuffled indices.
            # Forward pass
            y_pred = model(X_batch)
            # Compute loss
            loss = loss_fn(y_pred, t_batch)
            # Backward pass
            loss.backward()
            # Parameter update
            optimizer.step()
            # Accumulate loss
            epoch_loss += loss.item() * len(indices)
            count += len(indices)
        # Average loss calculation
        avg_loss = epoch_loss / count
        if epoch % print_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluate model performance.
    with torch.no_grad():                                                  # Disables gradient calculations during evaluation (saves memory and computation)
        y_train_pred = model(X_train)                                      # The model computes predictions for the entire training and test datasets.
        train_acc = accuracy(y_train_pred, t_train)
        y_test_pred = model(X_test)                                        # The accuracy function (defined elsewhere) computes the fraction of correct predictions
        test_acc = accuracy(y_test_pred, t_test)                           # for both training and test sets.

    # The model's effective_degree() method computes a weighted average of the polynomial degrees based on the learned gating parameters.
    # This gives insight into which degrees (or combination of degrees) the model finds most useful.
    effective_degree, gating_weights = model.effective_degree()
    print("\nLearned gating weights (for degrees 0 to max_M):")
    print(gating_weights)
    print(f"Effective polynomial degree: {effective_degree:.2f}")
    print(f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")
