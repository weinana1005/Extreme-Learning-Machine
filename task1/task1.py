import math
import torch
import torch.optim as optim


# ---------------------------
# Helper Functions
# ---------------------------

def compute_features(X, M):
    """
    Compute polynomial features for each row in X up to order M.
    Ordering:
      - Degree 0: constant term.
      - Degree 1: linear terms (in order).
      - Degree 2: first squares (in order), then interaction terms (i < j).
      - Degree 3: cubes, then (square * linear) terms in two orders, then triple interactions.
    Args:
        X (torch.Tensor): Input tensor of shape (N, D).
        M (int): Maximum polynomial order.
    Returns:
        torch.Tensor: Tensor of shape (N, p) where p is total number of polynomial terms.
    """
    N, D = X.shape                                                   # N = number of samples (rows)   D = number of input features per sample (columns)
    features = []                                                    # we’ll fill with different sets of polynomial features.

    # Degree 0
    features.append(torch.ones((N, 1), dtype=X.dtype, device=X.device))
    # We create a tensor of shape (N,1) filled with ones, which corresponds to the constant/bias term.
    # We append it to the features list. Now features[0] is a column of ones.

    if M >= 1:
        # Degree 1: linear terms xi
        features.append(X)

    # If M is at least 1, we include the linear terms (the original features x1, x2,..., xd)
    # Here, X is shape (N, D).

    if M >= 2:
        # Degree 2: squares xi^2 + pairwise interactions xixj
        squares = X ** 2                                             # This yields shape (N, D) again, but now each column is xi^2
        features.append(squares)
        # Cross (interaction) terms: for each i < j
        cross_terms = []
        for i in range(D):
            for j in range(i + 1, D):
                cross_terms.append((X[:, i] * X[:, j]).unsqueeze(1)) # Unsqueeze: change from (N,) to (N, 1)
        if cross_terms:
            features.append(torch.cat(cross_terms, dim=1))           # Combine all tensors along d=1 into one big matrix.

    if M >= 3:
        # Degree 3: cubes xi^3 + mixed terms xi^2xj + triple products xixjxk
        cubes = X ** 3
        features.append(cubes)
        # Terms: for each i < j, x_i^2 * x_j
        mix1 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix1.append((X[:, i] ** 2 * X[:, j]).unsqueeze(1))
        if mix1:
            features.append(torch.cat(mix1, dim=1))
        # Terms: for each i < j, x_i * x_j^2
        mix2 = []
        for i in range(D):
            for j in range(i + 1, D):
                mix2.append((X[:, i] * X[:, j] ** 2).unsqueeze(1))
        if mix2:
            features.append(torch.cat(mix2, dim=1))
        # Triple interactions: for i < j < k
        triple = []
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    triple.append((X[:, i] * X[:, j] * X[:, k]).unsqueeze(1))
        if triple:
            features.append(torch.cat(triple, dim=1))

    return torch.cat(features, dim=1)
    # we concatenate all lists in features horizontally (along dim=1) into a single 2D tensor of shape (N, P)
    # P = total number of polynomial terms for degree up to M.


def logistic_fun(w, M, X):
    """
    Compute logistic function y = σ(f_M(X; w)) for input X.
    f_M(X; w) is computed as the dot product between polynomial features (up to order M) and weights w.

    Args:
        w (torch.Tensor): Weight vector of shape (p,).
        M (int): Polynomial order.
        X (torch.Tensor): Input tensor of shape (N, D).

    Returns:
        torch.Tensor: Predicted probabilities, tensor of shape (N,).
    """
    features = compute_features(X, M)  # shape: (N, p)
    f = features.matmul(w)  # .matmul computes the dot product for each samples in feature. (feature dot w)
    return torch.sigmoid(f)


def accuracy(y_pred, t):
    """
    Compute classification accuracy given predicted probabilities and true binary targets.

    Args:
        y_pred (torch.Tensor): Predicted probabilities (N,).
        t (torch.Tensor): True binary labels (N,).

    Returns:
        float: Accuracy value.
    """
    preds = (y_pred >= 0.5).float()                 # Tensor of binary predictions (1.0 or 0.0) based on the 0.5 threshold.
    return (preds == t).float().mean().item()       # Compare the binary predictions with the true labels.
                                                    # Convert the results of the comparison into floats (1.0 for correct, 0.0 for incorrect) and compute the mean, which gives the accuracy.
                                                    # Return the accuracy from tensor to Python float (Scalars).


# ---------------------------
# Loss Classes
# ---------------------------

class MyCrossEntropy:
    """
    Loss class for computing binary cross-entropy loss.
    """

    def __call__(self, y_pred, t):
        eps = 1e-7  # Avoid numerical issues
        y_pred = torch.clamp(y_pred, eps, 1 - eps) #  ensures that every element in y_pred is within the range [eps, 1-eps]
        loss = - (t * torch.log(y_pred) + (1 - t) * torch.log(1 - y_pred))
        return loss.mean()

# y-pred = tensor of predicted probabilities (each value between 0 and 1).
# t = tensor of the true binary labels (0 or 1).
# Formula: https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning


class MyRootMeanSquare:
    """
    Loss class for computing root-mean-square error (RMSE) loss.
    """

    def __call__(self, y_pred, t):
        loss = torch.sqrt(torch.mean((y_pred - t) ** 2))
        return loss

# Formula: https://c3.ai/glossary/data-science/root-mean-square-error-rmse/

# ---------------------------
# SGD Optimisation Function
# ---------------------------

def fit_logistic_sgd(X, t, M, loss_fn, learning_rate=0.01, batch_size=32, epochs=100):
    """
    Stochastic minibatch gradient descent for logistic regression.

    Args:
        X (torch.Tensor): Input tensor of shape (N, D).
        t (torch.Tensor): Binary targets tensor of shape (N,).
        M (int): Polynomial order for feature expansion.
        loss_fn: Loss function instance (MyCrossEntropy or MyRootMeanSquare).
        learning_rate (float): Learning rate for SGD.
        batch_size (int): Mini-batch size.
        epochs (int): Number of epochs.

    Returns:
        torch.Tensor: Optimised weight vector (p,).
    """
    N, D = X.shape
    p = compute_features(X, M).shape[1]                     # Apply polynomial expansion, and get its column number

    w = torch.zeros(p, requires_grad=True)                  # creates a weight vector w of shape (p,), initialized to 0.
                                                            # requires_grad=True will have its gradients computed automatically during backpropagation (.backward)
    optimizer = optim.SGD([w], lr=learning_rate)    # creates an SGD optimizer that will update w using specific lr.

    # Ensure at least 10 printed updates (including first and last epoch)
    print_interval = max(1, epochs // 10)                   # This calculates how frequently (in terms of epochs) the function will print the loss.
                                                            # ensures that there are at least 10 updates printed during the training process,
                                                            # or at least one if the number of epochs is low.

    # Training Loop
    for epoch in range(epochs):
        permutation = torch.randperm(N)                     # Random permutation of indices to shuffle the data at the start of each epoch.
        epoch_loss = 0.0                                    # Accumulates the loss over all mini-batches in the epoch.
        count = 0                                           # # keeps track of the total number of samples processed in the epoch.
        # Mini-batch loop: Iterates over the dataset in chunks of batch_size
        for i in range(0, N, batch_size):                   # Ex: if N is 200 and batch_size is 32, then i will take values 0, 32, 64, and so on.
            optimizer.zero_grad()   # Clears previous gradient values.
            indices = permutation[i:i + batch_size]         # Indices will be from i to i + batch size
            # Extract Mini-Batch Data
            X_batch = X[indices]                            # subset of the input data for this mini-batch.
            t_batch = t[indices]                            # contains the corresponding binary target labels.
            # Forward Pass
            y_pred = logistic_fun(w, M, X_batch)            # computes the predicted probabilities for the mini-batch using the logistic regression model.
            # Compute loss
            loss = loss_fn(y_pred, t_batch)                 # The chosen loss function (passed as loss_fn) is applied to the predictions y_pred and the true labels t_batch.
                                                            # For binary classification, this could be cross-entropy or RMSE, depending on which loss function instance you provided.
            # Backpropagation
            loss.backward()                                 # This computes the gradients of the loss with respect to the model parameters (here, the weight vector w)
                                                            # The gradients are automatically stored in w.grad
            # Update Weights
            optimizer.step()                                # The optimizer (SGD in this case) updates the weight vector w  based on the computed gradients and the specified learning rate.
                                                            # This is where the model "learns" by adjusting its parameters.
            # Accumulate Loss
            epoch_loss += loss.item() * len(indices)        # This line accumulates the total loss for the epoch.
            # Count Samples Processed
            count += len(indices)
        # Average loss calculation
        avg_loss = epoch_loss / count
        if epoch % print_interval == 0 or epoch == epochs - 1:  # ensures that the loss is printed (e.g., every 10% of the total epochs).
                                                                # ensures that the final epoch's loss is always printed
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}") # shows which epoch is running and the average loss for that epoch
    return w.detach()                                        # Return the optimized weight vector after training is complete.


# ---------------------------
# Data Generation Functions
# ---------------------------

def generate_underlying_weights(M, D):
    """
    Generate underlying weight vector for data generation given M and D.
    For data generation, M is fixed to 2.
    Underlying weights are defined as:
      w[i] = (-1)^(k) * sqrt(k) / p, where k = p - i and p = total number of polynomial terms.

    Args:
        M (int): Polynomial order.
        D (int): Dimensionality of input.

    Returns:
        torch.Tensor: Weight vector of shape (p,).
    """
    # Determine the Total Number of Features (p)
    p = compute_features(torch.zeros(1, D), M).shape[1]         # To compute the total number of polynomial features generated from an input with dimensionality
    # Initialize the Weight Vector
    w_true = torch.zeros(p)                                     # shape (p,). meaning there is one weight per polynomial feature
    # Populate the Weight Vector
    for i in range(p):
        k = p - i
        w_true[i] = ((-1) ** k) * math.sqrt(k) / p              # weight Formula
    return w_true                                               # sequence of weights


def generate_dataset(N, D, w_true, noise_std=1.0):
    """
    Generate dataset of N examples in D dimensions.
    Each input x is uniformly sampled from [-5, 5]^D.
    The true probability is computed using logistic_fun with fixed M=2 and w_true.
    A Gaussian noise is added and then thresholded at 0.5 to get binary target.

    Args:
        N (int): Number of examples.
        D (int): Input dimensionality.
        w_true (torch.Tensor): Underlying weight vector (for M=2).
        noise_std (float): Standard deviation of added Gaussian noise.

    Returns:
        X (torch.Tensor): Tensor of shape (N, D).
        t (torch.Tensor): Binary targets tensor of shape (N,). Containing 0s and 1s.
    """
    # Uniformly sample inputs from [-5, 5]
    X = (10 * torch.rand(N, D)) - 5                             # Generates a tensor of shape (N, D) with values uniformly between 0 & 1. Multiply by 10 then -5.
    # Compute true probability using M=2 for data generation
    y_true = logistic_fun(w_true, 2, X)                     # w = w_true, m = 2, X = X
    # Add Gaussian noise and threshold
    noise = torch.normal(mean=0.0, std=noise_std, size=y_true.shape)
    y_noisy = y_true + noise                                    # Generate noise with same size of y_true and add together
    t = (y_noisy >= 0.5).float()                                # return 1.0 (true) if at least 0.5, else 0 (False)
    return X, t


# ---------------------------
# Main Task Script
# ---------------------------

if __name__ == "__main__":
    torch.manual_seed(42)                                       # random seed

    # Settings for data generation
    D = 5
    N_train = 200
    N_test = 100
    M_data = 2                                                   # For generating data, we use M=2

    # Generate underlying weights for data generation
    w_true = generate_underlying_weights(M_data, D)
    print("Underlying weights for data generation (M=2):")
    print(w_true)

    # Generate training and test sets
    X_train, t_train = generate_dataset(N_train, D, w_true)
    X_test, t_test = generate_dataset(N_test, D, w_true)

    # Define loss functions (__call__ function)
    cross_entropy_loss = MyCrossEntropy()
    rmse_loss = MyRootMeanSquare()

    # Hyperparameter values for M to test and SGD settings
    M_values = [1, 2, 3]
    learning_rate = 0.01
    batch_size = 32
    epochs = 100

    # For storing results
    results = {}

    print("\n--- Training with Cross-Entropy Loss ---")
    for M in M_values:
        print(f"\nTraining model with polynomial order M = {M} (Cross-Entropy)")
        w_opt = fit_logistic_sgd(X_train, t_train, M, cross_entropy_loss, learning_rate, batch_size, epochs)            # Expands the features up to order M. Initializes weights to zero.
                                                                                                                        # Runs mini-batch SGD for 100 epochs, printing loss periodically.
                                                                                                                        # w_opt = The final learned weights for that polynomial order.
        # Evaluate on training set
        y_train_pred = logistic_fun(w_opt, M, X_train)                                                                  # Produces predicted probabilities for each sample.
        train_acc = accuracy(y_train_pred, t_train)
        # Evaluate on test set
        y_test_pred = logistic_fun(w_opt, M, X_test)
        test_acc = accuracy(y_test_pred, t_test)
        results[f"CE_M{M}"] = {"train_accuracy": train_acc, "test_accuracy": test_acc}                                  # Records the final accuracies.
        print(f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")

    print("\n--- Training with RMSE Loss ---")
    for M in M_values:                                                                                                  # Same process for rmse_loss
        print(f"\nTraining model with polynomial order M = {M} (RMSE)")
        w_opt = fit_logistic_sgd(X_train, t_train, M, rmse_loss, learning_rate, batch_size, epochs)
        # Evaluate on training set
        y_train_pred = logistic_fun(w_opt, M, X_train)
        train_acc = accuracy(y_train_pred, t_train)
        # Evaluate on test set
        y_test_pred = logistic_fun(w_opt, M, X_test)
        test_acc = accuracy(y_test_pred, t_test)
        results[f"RMSE_M{M}"] = {"train_accuracy": train_acc, "test_accuracy": test_acc}
        print(f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")

print("\n----------------------------------")
def main():
    # Questions and answers as tuples: (question_text, answer_text)
    questions_and_answers = [
        (
            "Q1) Consider what a metric (other than the losses) is appropriate for this classification problem. "
            "Justify briefly your choice of metric in a printed comment (no more than 50 words).",
            """A1) Accuracy is straightforward for binary classification, measuring the fraction of correct predictions.
    It directly reflects how often the model’s predicted label matches the true label,
    making it easy to interpret and compare across different polynomial orders and loss functions."""
        ),
        (
            "Q2) For each loss, report the metric using printed messages a) on the model prediction and b) on "
            "observed training data, both with respect to the underlying 'true' classes as ground-truth.",
            "A2) Cross-Entropy Loss:\n"
            "   - M=1: Train Acc = 61.5%, Test Acc = 44.0%\n"
            "   - M=2: Train Acc = 68.5%, Test Acc = 56.0%\n"
            "   - M=3: Train Acc = 69.0%, Test Acc = 58.0%\n\n"
            "   RMSE Loss:\n"
            "   - M=1: Train Acc = 59.0%, Test Acc = 46.0%\n"
            "   - M=2: Train Acc = 72.0%, Test Acc = 56.0%\n"
            "   - M=3: Train Acc = 85.5%, Test Acc = 44.0%"
        ),
        (
            "Q3) Comment briefly on the difference between the two in a printed message (no more than 100 words).",
            """A3) Cross-entropy aligns better with probability outputs for classification,
    often leading to more stable test performance.
    RMSE can push probabilities away from 0 or 1,
    which may result in higher training accuracy but poorer generalization.
    Overall, cross-entropy yields better-calibrated probabilities for binary classification."""
        )
    ]

    # Print questions and answers with a blank line between each
    for question, answer in questions_and_answers:
        print(question)
        print(answer)
        print()  # Blank line for readability


if __name__ == "__main__":
    main()

