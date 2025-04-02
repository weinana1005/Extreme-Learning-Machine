import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os

# ---------------------------
# Helper Function: evaluate_model
# ---------------------------
def evaluate_model(model, testloader, criterion, device):
    """
    Evaluate the model on the test set.
    Returns:
        avg_loss (float): Average test loss.
        test_acc (float): Test accuracy percentage.
        test_f1 (float): F1-score.
    """
    model.eval()
    total, correct = 0, 0
    running_test_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    avg_loss = running_test_loss / len(testloader)
    test_acc = correct / total * 100

    # Compute F1-score
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    test_f1 = compute_f1(all_outputs, all_labels)

    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {test_acc:.2f}% | Test F1-score: {test_f1:.4f}")
    return avg_loss, test_acc, test_f1



# ---------------------------
# MyExtremeLearningMachine
# ---------------------------
class MyExtremeLearningMachine(nn.Module):
    def __init__(self, num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1):     # Initialize the model with parameters
        """
        ELM with one fixed convolutional layer and one trainable FC layer.
        Args:
            num_channels (int): Number of input channels (e.g., 3 for RGB).
            hidden_maps (int): Number of convolutional feature maps.
            num_classes (int): Number of output classes.
            kernel_size (int): Convolution kernel size. (e.g. 3x3)
            padding = 1: Adds a 1-pixel border around the input, which helps maintain the size of input.
                         without padding, size might be reduced form 32x32 to 30x30
            std (float): Std. for Gaussian initialization of fixed weights.
        """
        super(MyExtremeLearningMachine, self).__init__()

        # Fixed convolutional layer (nn.Conv2d): for feature extraction
        # During training only the parameters of the FC layer would be updated (since the convolutional layer’s parameters are frozen).
        self.fixed_conv = nn.Conv2d(num_channels, hidden_maps, kernel_size, padding=1)  # This layer is meant to extract features from the input image.
        self.initialise_fixed_layers(std)                                               # initialize the weights [N~(0,std^2)] of the convolutional layer.

        # Frozen fixed conv weights
        for param in self.fixed_conv.parameters():                                      # Loop over all the parameters (weight & biases) of the fixed convolutional layer.
            param.requires_grad = False                                                 # Setting this to False tells PyTorch not to compute gradients for that parameter during the backward pass.

        # Fully-connected layer (trainable): for classification
        # CIFAR-10 images: 32x32; output shape of conv: (hidden_maps, 32, 32)
        self.fc = nn.Linear(hidden_maps * 32 * 32, num_classes)                         # self.fc is defined to take an input of size (hidden_maps * 32 * 32) and produce an output vector size (num_classes).
                                                                                        # Takes the flattened output of the convolutional layer and maps it to the output classes. This layer is trainable.

    # Initialise the weights and biases of the convolution layer (self.fixed_conv)
    def initialise_fixed_layers(self, std):
        """
        Initialise the convolutional kernels with a Gaussian (mean=0, given std).
        """
        nn.init.normal_(self.fixed_conv.weight, mean=0.0, std=std)                      # In-place function that fills the given tensor with values drawn from a normal distribution
        if self.fixed_conv.bias is not None:                                            # If not None, set all bias values to 0.
            nn.init.constant_(self.fixed_conv.bias, 0.0)

    def forward(self, x):
        # Fixed conv layer (non-trainable)
        x = self.fixed_conv(x)                          # shape: (N, hidden_maps, 32, 32)
                                                        # This layer extracts features using pre-initialized filters (which are not updated during training).
        x = F.relu(x)                                   # ReLU activation function is applied to introduce non-linearity
        # Flatten
        x = x.view(x.size(0), -1)                       # To pass the FC layer, tensor must be reshape(flattened) into a 2D tensor.
                                                        # change from ((N, hidden_maps, 32, 32)) to (N, hidden_maps x 32 x 32)
        # FC layer
        x = self.fc(x)                                  # The flattened tensor is passed through the FC layer (self.fc).
                                                        # The FC layer performs a linear transformation, mapping the high-dimensional vector to the number of output classes
        return x                                        # shape: (N, num_classes)


# ---------------------------
# MyMixUp for Data Augmentation
# ---------------------------
# The MyMixUp class applies the mixup augmentation,
# which creates new training samples by combining pairs of images and their labels in a convex combination.
class MyMixUp:
    def __init__(self, alpha=0.2, seed=42):                                              # Higher alpha makes the mixup more even (closer to 0.5)
        """
        Mixup augmentation.
        Args:
            alpha (float): Beta distribution parameter.
            seed (int): For reproducibility.
        Technique:
            one-hot encoded label: use to represent categorical labels in a format that machine learning models can process.
        """
        self.alpha = alpha
        np.random.seed(seed)

    def __call__(self, images, labels):
        batch_size = images.size(0)                                                     # Returns the number of images in the batch
        lam = np.random.beta(self.alpha, self.alpha)                                    # draws a sample from the Beta distribution with parameters alpha and alpha.
                                                                                        # result: scalar in range [0,1], determines how much of one image versus another will contribute to the new mixed image.
                                                                                        # e.g. lam=0.7 means 70% one image, 30% another

        # Generate a Random Permutation of the Batch Indices
        index = torch.randperm(batch_size)                                              # random permutation of indices from 0 to batch_size - 1

        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[index, :]                      # lam * images: scales each pixel of the original images.
                                                                                        # (1 - lam) * images[index, :]: scales each pixel of the randomly paired images.
                                                                                        # The result is a new tensor mixed_images of the same shape as images, where each image is a linear combination of two images.

        # Convert labels to one-hot
        num_classes = 10                                                                # Sets the number of classes. For CIFAR‑10, there are 10 classes.
        labels_onehot = torch.zeros(batch_size, num_classes, device=images.device)      # Creates a tensor of shape (batch_size, 10) filled with zeros.
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)                                # Reshapes the labels tensor from shape (batch_size,) to (batch_size, 1). This makes it compatible for the scatter operation.
                                                                                        # After executing this line, labels_onehot becomes a one-hot encoded tensor. Each row has a 1 in the column corresponding to the class label, and 0s elsewhere.
        labels_shuffled = torch.zeros(batch_size, num_classes, device=images.device)    # Same process for shuffled version
        labels_shuffled.scatter_(1, labels[index].view(-1, 1), 1)
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled                # This line computes a convex combination of the original one-hot encoded labels (labels_onehot) and the shuffled one-hot encoded labels (labels_shuffled)
        return mixed_images, mixed_labels                                               # Return the mixed images and their corresponding mixed labels (Will be used during training)


# ---------------------------
# MyEnsembleELM for Model Ensemble
# ---------------------------
class MyEnsembleELM:
    def __init__(self, num_models=3, seed=42, **elm_kwargs):
        """
        Ensemble of ELM models.
        Combines multiple instances of MyExtremeLearningMachine, each with individually
        initialized fixed weights. The initialization is seeded for reproducibility.

        Args:
            num_models (int): Number of models to ensemble. Must be between 1 and 10.
            seed (int): Seed for reproducibility. Each model's seed is offset to ensure different initialization.
            elm_kwargs: Keyword arguments for MyExtremeLearningMachine.
                        It's like a dictionary. Holds all the extra parameters that you want to pass to the MyExtremeLearningMachine constructor when creating each model in the ensemble.
                        When you need parameters like "num_channels", "hidden_maps"... Use unpacking operator (**elm_kwargs), and pass them to MyExtremeLearningMachine.

        Raises:
            ValueError: If num_models is not within the acceptable range.
        """
        # Check hyperparameter range.
        if num_models < 1 or num_models > 10:
            raise ValueError("num_models must be between 1 and 10.")
        self.num_models = num_models
        self.models = []                                                                # Store the individual ELM models.
        # Initialize each model with an offset seed for reproducibility.
        for i in range(num_models):
            torch.manual_seed(seed + i)                                                 # Each model gets a unique but reproducible seed.
            model = MyExtremeLearningMachine(**elm_kwargs)
            self.models.append(model)                                                   # Model added into the list with different initialization.

    def predict(self, x):
        """
        Combine predictions from all ensemble models by averaging.
        Args:
            x (torch.Tensor): Input tensor. Shape (N,...)
        Returns:
            torch.Tensor: Averaged predictions from the ensemble.
        """
        preds = [model(x) for model in self.models]                                     # For each model, it passes the input x through the model (i.e., calls the model’s forward method), producing a prediction tensor. Shape (N, num_classes)
        return torch.mean(torch.stack(preds), dim=0)                                    # torch.stack converts the list of prediction tensors into a single tensor.
                                                                                        # If there are M models and each output has shape (N, num_classes), the stacked tensor will have shape (M, N, num_classes).
                                                                                        # torch.mean computes the average across the first dimension (dimension 0), resulting in a final tensor of shape (N, num_classes).
                                                                                        # This averaged output represents the ensemble's combined prediction. (With the same batch size and number of classes as the individual outputs.)
                                                                                        # This will help reduce variance and can improve the overall robustness and accuracy of predictions.


# ---------------------------
# Helper Functions
# ---------------------------
def compute_accuracy(outputs, labels):                                                    # computes the maximum value along dimension 1 of the outputs tensor.
    # Finding the predicted class
    _, predicted = torch.max(outputs, 1)                                                  # computes the maximum value along dimension 1 (i.e., across the class scores for each sample).
                                                                                          # _ is used since we only want the indices, not the max value.
    correct = (predicted == labels).sum().item()                                          # Counts how many predictions match the true labels and converts the sum to a standard Python number.
    return correct / labels.size(0)                                                       # Divides the count of correct predictions by the total number of samples in the batch, giving the accuracy.


# ---------------------------
# Compute Macro F1-score
# ---------------------------
def compute_f1(outputs, labels, num_classes=10):
    """
    Compute macro F1-score for multi-class classification.
    """
    _, predicted = torch.max(outputs, 1)  # Get predicted class indices

    f1_scores = []
    for cls in range(num_classes):
        true_positive = ((predicted == cls) & (labels == cls)).sum().item()
        false_positive = ((predicted == cls) & (labels != cls)).sum().item()
        false_negative = ((predicted != cls) & (labels == cls)).sum().item()

        precision = true_positive / (true_positive + false_positive + 1e-8)  # Avoid division by zero
        recall = true_positive / (true_positive + false_negative + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes  # Compute macro F1-score


# This function helps you quickly assess how well your model is performing by visually comparing predictions to ground truth across several examples.
def create_montage(images, labels, predictions, classes, save_path="result.png"):
    """
    Create and save a montage of images with ground-truth and predicted labels.
    Assumes images are normalized (mean=0.5, std=0.5). And they need to be
    denormalized before visualization.
    """
    # Denormalize the Images
    images = images * 0.5 + 0.5                                                           # Denormalize to [0,1]

    # Convert to a NumPy Array
    images_np = images.cpu().numpy()                                                      # Moves the tensor to the CPU and converts it into a NumPy array for further processing with libraries like PIL

    # Create a List of PIL Images with Overlaid Text
    montage_list = []
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # or 6, if you want even smaller
    except IOError:
        print("TTF font not found; using default font instead (may still appear large).")
        font = ImageFont.load_default()

    for i in range(images_np.shape[0]):                                                   # Loop over each image in the batch
        img = np.transpose(images_np[i],(1, 2, 0)) * 255                             # Transpose from (channels, height, width) to (height, width, channels) and scale to [0,255]
        img = img.astype(np.uint8)                                                        # Converts the array to unsigned 8-bit integers.
        pil_img = Image.fromarray(img)                                                    # Converts the NumPy array to a PIL image.

        draw = ImageDraw.Draw(pil_img)                                                    # A drawing object is created using ImageDraw.Draw(pil_img)
        text = f"GT: {classes[labels[i]]}\nP: {classes[predictions[i]]}"                  # GT: <ground-truth label>, P: <predicted label>
        draw.text((5, 5), text, fill=(255, 0, 0), font=font)                           # Overlays the text at position (5,5) in red color.
        montage_list.append(pil_img)

    # Determine Grid Size
    grid_size = int(np.sqrt(len(montage_list)))                                           # Calculate grid size using square root of the total number of images in the montage list.

    # Get Image Dimensions
    width, height = montage_list[0].size                                                  # Gets the width and height from an individual image.

    # Create Montage Canvas
    montage_img = Image.new('RGB', (width * grid_size, height * grid_size))    # Creates a new blank image (canvas) large enough to hold all images in the grid.

    # Paste Images into the Montage
    for idx, img in enumerate(montage_list):                                              # Loop over each image in the montage list.
        row = idx // grid_size                                                            # Calculate row index.
        col = idx % grid_size                                                             # Calculate column index.
        montage_img.paste(img, (col * width, row * height))                          # Pastes the image at the correct position.
    montage_img.save(save_path)                                                           # Saves the montage image to the specified file path.
    print(f"Montage saved as {save_path}")                                                # Print a confirmation message.


# The purpose of create_mixup_montage is to visualize the results of the MixUp augmentation.
# It takes a batch of mixup-augmented images, denormalizes them, and arranges them into a grid (montage) so you can see how the images have been blended.
# This helps you verify that the mixup process is working as intended before using the augmented images in training.
def create_mixup_montage(images, save_path="mixup.png", grid_size=4):                     # Similar process as "create_montage". See Above.
    """
    Create and save a montage of mixup augmented images.
    Assumes images are normalized (mean=0.5, std=0.5).
    """
    images = images * 0.5 + 0.5  # Denormalize to [0,1]
    images_np = images.cpu().numpy()
    montage_list = []
    num_images = min(images_np.shape[0], grid_size * grid_size)
    for i in range(num_images):
        img = np.transpose(images_np[i], (1, 2, 0)) * 255
        img = img.astype(np.uint8)
        pil_img = Image.fromarray(img)
        montage_list.append(pil_img)
    width, height = montage_list[0].size
    montage_img = Image.new('RGB', (width * grid_size, height * grid_size))
    for idx, img in enumerate(montage_list):
        row = idx // grid_size
        col = idx % grid_size
        montage_img.paste(img, (col * width, row * height))
    montage_img.save(save_path)
    print(f"MixUp montage saved as {save_path}")


# ---------------------------
# Fit ELM using SGD
# ---------------------------
def fit_elm_sgd(model, trainloader, criterion, optimizer, epochs, device='cpu'):
    """
    This function trains an Extreme Learning Machine (ELM) model using stochastic minibatch gradient descent (SGD).
    It iterates over the training data for a given number of epochs, updating the model's trainable parameters based on the computed gradients.
    Args:
        model (nn.Module): An instance of MyExtremeLearningMachine.
        trainloader (DataLoader): DataLoader for the training dataset.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer (e.g., optim.SGD(model.parameters(), lr=0.001)).
        epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu' or 'cuda').
    Returns:
        model (nn.Module): The trained model.
    """
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()                                                                     # Sets the model to training mode for dropout and gradient calculation.
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)                         # Moves the inputs and labels to the specific device
            optimizer.zero_grad()                                                         # Clears the gradients from the previous mini-batch.
            outputs = model(inputs)                                                       # Computes the model’s predictions for the current batch.
            loss = criterion(outputs, labels)                                             # Calculates the loss between the predictions and the true labels using the provided loss function.
            loss.backward()                                                               # Computes the gradients of the loss with respect to the model's trainable parameters.
            optimizer.step()                                                              # Updates the model parameters based on the computed gradients.
            running_loss += loss.item()                                                   # Adds the scalar loss for the current batch to the running loss.
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
    return model


# ---------------------------
# Fit ELM using Direct Least-Squares Solver
# ---------------------------
def fit_elm_ls(model, trainloader, device='cpu', max_batches=50):
    """
    Optimize the ELM model using a direct least-squares solver for the fully-connected (FC) layer.
    This function extracts features from the training data using the fixed convolutional layer, then computes
    the optimal FC layer weights via the Moore-Penrose pseudoinverse.
    Args:
        model (nn.Module): An instance of MyExtremeLearningMachine.
        trainloader (DataLoader): DataLoader for the training dataset.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        max_batches (int, optional): Maximum number of batches to process to limit memory usage.
    Returns:
        model (nn.Module): The model with updated FC layer weights.
    """
    model.to(device)
    model.eval()                                                                                                        # Ensure model is in evaluation mode

    # Recompute the correct feature dimension using a dummy input
    dummy_input = torch.zeros(1, 3, 32, 32).to(device)                                                                  # Assumes 3 channels, 32x32 image
    dummy_features = model.fixed_conv(dummy_input)                                                                      # This input is passed through the fixed convolutional layer and ReLU to determine the actual feature dimensions.
    dummy_features = F.relu(dummy_features)
    dummy_features = dummy_features.view(dummy_features.size(0), -1)
    p = dummy_features.size(1)
    num_classes = model.fc.out_features                                                                                 # The number of output classes (num_classes) is retrieved from the current FC layer.

    # Update the FC layer to have the correct input dimension.
    model.fc = nn.Linear(p, num_classes).to(device)                                                                     # The FC layer is redefined with the correct input dimension p (which may differ when using different hyperparameters, such as kernel size or hidden maps).

    feature_list = []
    label_list = []                                                                                                     
    batch_count = 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            # Pass inputs through fixed conv layer and ReLU, then flatten.
            features = model.fixed_conv(inputs)
            features = F.relu(features)
            features = features.view(features.size(0), -1)  # shape: (batch_size, p)
            feature_list.append(features)
            # Convert labels to one-hot encoding.
            batch_size = labels.size(0)
            one_hot = torch.zeros(batch_size, num_classes, device=device)
            one_hot.scatter_(1, labels.view(-1, 1).to(device), 1)
            label_list.append(one_hot)
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break
    X = torch.cat(feature_list, dim=0)      # shape: (N, p)
    Y = torch.cat(label_list, dim=0)        # shape: (N, num_classes)

    # Compute the Moore-Penrose pseudoinverse of X.
    X_pinv = torch.linalg.pinv(X)           # shape: (p, N)
    # Compute optimal FC weights: W_opt = X_pinv @ Y, shape: (p, num_classes)
    W_opt = X_pinv @ Y
    # Update the FC layer of the model with the computed weights.
    with torch.no_grad():
        model.fc.weight.copy_(W_opt.t())    # FC weights shape: (num_classes, p), so we transpose W_opt.
        if model.fc.bias is not None:
            model.fc.bias.zero_()           # Set bias to zero.
    return model



# ---------------------------
# Random Hyperparameter Search using fit_elm_ls
# ---------------------------
def random_hyperparameter_search(trainloader, testloader, device, num_trials=5):
    """
    Perform a random hyperparameter search using fit_elm_ls.
    This function tries different hyperparameter settings (hidden_maps, std, kernel_size),
    applies the direct least-squares solver to optimize the ELM model, evaluates on the test set,
    and returns the best model and its hyperparameters.
    """
    best_acc = 0.0
    best_f1 = 0.0  # Track the best F1-score
    best_params = None
    best_model = None
    criterion = nn.CrossEntropyLoss()

    for trial in range(num_trials):
        # Randomly sample hyperparameters.
        hidden_maps = np.random.choice([16, 32, 64])
        std = np.random.uniform(0.05, 0.2)
        kernel_size = np.random.choice([3, 5])
        print(f"Trial {trial+1}: hidden_maps={hidden_maps}, std={std:.3f}, kernel_size={kernel_size}")

        # Create a new model with sampled hyperparameters.
        model_trial = MyExtremeLearningMachine(
            num_channels=3, hidden_maps=hidden_maps, num_classes=10, kernel_size=kernel_size, std=std
        )
        model_trial = fit_elm_ls(model_trial, trainloader, device=device, max_batches=50)

        # Now correctly expect three return values
        test_loss, test_acc, test_f1 = evaluate_model(model_trial, testloader, criterion, device)

        print(f"Trial {trial+1} Test Accuracy: {test_acc:.2f}% | F1-score: {test_f1:.4f}")

        # Track the best model based on accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1  # Track best F1-score too
            best_params = (hidden_maps, std, kernel_size)
            best_model = model_trial

    return best_model, best_params, best_acc, best_f1


# ---------------------------
# Helper function to evaluate ensemble models
# ---------------------------
def evaluate_ensemble(model, dataloader, device):
    for sub_model in model.models:
        sub_model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.predict(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    avg_loss = running_loss / len(dataloader)
    acc = correct / total * 100
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    f1 = compute_f1(all_outputs, all_labels)
    return avg_loss, acc, f1



# ---------------------------
# Main Task Script for Task 2a
# ---------------------------
def main():
    # This section sets up the data transformations for the CIFAR‑10 dataset.
    transform = transforms.Compose([                                                                                    # Chains several image transformations
        transforms.ToTensor(),                                                                                          # Converts images to PyTorch tensors with pixel values in the range [0, 1].
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                                # Normalizes each channel of the image. For each pixel in the 3 channels, it subtracts 0.5 and divides by 0.5, mapping [0, 1] to [-1, 1].
                                                                                                                        # Formula: Output = (Input - mean) / std
    ])
    batch_size = 64                                                                                                     # number of images per batch to 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)              # train=True flag loads the training set, download=True ensures the dataset is downloaded
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)              # train=False loads the test set.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)             # Wraps the training dataset in a DataLoader that divides the data into batches of size 64.
                                                                                                                        # shuffle=True randomizes the order of the training samples each epoch, which helps improve training performance.
                                                                                                                        # num_workers=2 specifies the number of subprocesses to use for data loading.
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)              # Similar process, without shuffling.
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')                          # Defines a tuple containing the names of the 10 classes in the CIFAR‑10 dataset. These names are used later for visualization and for interpreting predictions.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                               # If a GPU is available, torch.device("cuda") is used; otherwise, it falls back to the CPU.

    ##########################################
    # Compare Speed & Performance: Ensemble ELM trained with SGD vs. LS
    ##########################################
    print("Training Ensemble ELM using SGD...")
    ensemble_sgd = MyEnsembleELM(num_models=3, num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1)
    ensemble_sgd_optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for model in ensemble_sgd.models]
    ensemble_epochs = 20
    sgd_start = time.time()

    # === ADD: Checkpoint logic for Ensemble ELM (SGD) ===
    ensemble_sgd_ckpt_path = "ensemble_sgd_checkpoint.pth"
    history_ensemble_sgd = []
    start_epoch_ensemble_sgd = 0

    if os.path.exists(ensemble_sgd_ckpt_path):
        ckpt = torch.load(ensemble_sgd_ckpt_path, map_location=device)
        start_epoch_ensemble_sgd = ckpt['epoch']
        # Load each model + optimizer
        for i, sub_model in enumerate(ensemble_sgd.models):
            sub_model.load_state_dict(ckpt[f'model_{i}_state_dict'])
            ensemble_sgd_optimizers[i].load_state_dict(ckpt[f'optimizer_{i}_state_dict'])
        # If there's a history, load + print it
        if 'history' in ckpt:
            history_ensemble_sgd = ckpt['history']
            print("==== Loaded Ensemble SGD training history from checkpoint ====")
            for entry in history_ensemble_sgd:
                print(f"Epoch {entry['epoch']}: "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry['train_f1']:.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print(f"Loaded Ensemble SGD checkpoint from epoch {start_epoch_ensemble_sgd}")

    # Start training from loaded epoch
    for epoch in range(start_epoch_ensemble_sgd, ensemble_epochs):
        for idx, model_sub in enumerate(ensemble_sgd.models):
            model_sub.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                ensemble_sgd_optimizers[idx].zero_grad()
                outputs = model_sub(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                ensemble_sgd_optimizers[idx].step()

        train_loss, train_acc, train_f1 = evaluate_ensemble(ensemble_sgd, trainloader, device)
        test_loss, test_acc, test_f1 = evaluate_ensemble(ensemble_sgd, testloader, device)
        print(f"[Ensemble SGD] Epoch {epoch + 1}/{ensemble_epochs} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

        # Save in history
        history_ensemble_sgd.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

        # Save checkpoint
        ckpt_dict = {'epoch': epoch + 1, 'history': history_ensemble_sgd}
        for i, sub_model in enumerate(ensemble_sgd.models):
            ckpt_dict[f'model_{i}_state_dict'] = sub_model.state_dict()
            ckpt_dict[f'optimizer_{i}_state_dict'] = ensemble_sgd_optimizers[i].state_dict()
        torch.save(ckpt_dict, ensemble_sgd_ckpt_path)
        print(f"Checkpoint saved at {ensemble_sgd_ckpt_path}")

    sgd_ensemble_time = time.time() - sgd_start
    print(
        f"Total Ensemble SGD training time: {sgd_ensemble_time:.2f} sec "
        f"(avg {sgd_ensemble_time / ensemble_epochs:.2f} sec/epoch)")

    print("\nTraining Ensemble ELM using LS on the best model configuration (one pass per member)...")
    ensemble_ls = MyEnsembleELM(num_models=3, num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1)
    ls_start = time.time()

    # === ADD: Checkpoint logic for Ensemble ELM (LS) ===
    ensemble_ls_ckpt_path = "ensemble_ls_checkpoint.pth"
    ls_history = []
    ls_loaded = False

    if os.path.exists(ensemble_ls_ckpt_path):
        ckpt_ls = torch.load(ensemble_ls_ckpt_path, map_location=device)
        for i, sub_model in enumerate(ensemble_ls.models):
            sub_model.load_state_dict(ckpt_ls[f'model_{i}_state_dict'])
        if 'history' in ckpt_ls:
            ls_history = ckpt_ls['history']
            print("==== Loaded Ensemble LS training history from checkpoint ====")
            for entry in ls_history:
                print(f"Single-Pass LS => "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry['train_f1']:.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print("Loaded Ensemble LS checkpoint.")
        ls_loaded = True

    # Only run LS if not loaded
    if not ls_loaded:
        for idx, model_sub in enumerate(ensemble_ls.models):
            model_sub = fit_elm_ls(model_sub, trainloader, device=device, max_batches=50)
            ensemble_ls.models[idx] = model_sub
        ls_ensemble_time = time.time() - ls_start

        ls_train_loss, ls_train_acc, ls_train_f1 = evaluate_ensemble(ensemble_ls, trainloader, device)
        ls_test_loss, ls_test_acc, ls_test_f1 = evaluate_ensemble(ensemble_ls, testloader, device)
        print(
            f"Ensemble LS -> Train Loss: {ls_train_loss:.4f} | Train Acc: {ls_train_acc:.2f}% | Train F1: {ls_train_f1:.4f} "
            f"| Test Loss: {ls_test_loss:.4f} | Test Acc: {ls_test_acc:.2f}% | Test F1: {ls_test_f1:.4f}")
        print(f"Ensemble LS training time: {ls_ensemble_time:.2f} sec")

        # Save single pass to history
        ls_history.append({
            'train_loss': ls_train_loss,
            'train_acc': ls_train_acc,
            'train_f1': ls_train_f1,
            'test_loss': ls_test_loss,
            'test_acc': ls_test_acc,
            'test_f1': ls_test_f1
        })

        # Save LS checkpoint
        ls_ckpt_dict = {}
        for i, sub_model in enumerate(ensemble_ls.models):
            ls_ckpt_dict[f'model_{i}_state_dict'] = sub_model.state_dict()
        ls_ckpt_dict['history'] = ls_history
        torch.save(ls_ckpt_dict, ensemble_ls_ckpt_path)
        print(f"LS checkpoint saved at {ensemble_ls_ckpt_path}")

    # Evaluate final LS model
    ls_train_loss, ls_train_acc, ls_train_f1 = evaluate_ensemble(ensemble_ls, trainloader, device)
    ls_test_loss, ls_test_acc, ls_test_f1 = evaluate_ensemble(ensemble_ls, testloader, device)

    print("\nPerforming random hyperparameter search using fit_elm_ls (one pass each)...")
    best_model, best_params, best_acc, best_f1 = random_hyperparameter_search(trainloader, testloader, device,
                                                                              num_trials=5)
    print(f"Best Hyperparameters (hidden_maps, std, kernel_size): {best_params}")
    print(f"Best Test Accuracy from LS: {best_acc:.2f}% | F1-score: {best_f1:.4f}")

    criterion = nn.CrossEntropyLoss()
    print("\nApplying LS once on the best model to finalize weights...")
    best_model = fit_elm_ls(best_model, trainloader, device=device, max_batches=50)
    final_train_loss, final_train_acc, final_train_f1 = evaluate_model(best_model, trainloader, criterion, device)
    final_test_loss, final_test_acc, final_test_f1 = evaluate_model(best_model, testloader, criterion, device)
    print(f"[Best LS Model] Final -> Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%, "
          f"Test Loss: {final_test_loss:.4f}, Test Acc: {final_test_acc:.2f}%, F1-score: {final_test_f1:.4f}")

    best_model.eval()
    test_iter = iter(testloader)
    images, labels = next(test_iter)
    images, labels = images.to(device), labels.to(device)
    outputs = best_model(images)
    _, predicted = torch.max(outputs, 1)
    num_images = min(36, images.size(0))
    create_montage(images[:num_images], labels[:num_images].cpu().tolist(),
                   predicted[:num_images].cpu().tolist(), classes, save_path="new_result.png")
    print("Montage saved as new_result.png.")


if __name__ == '__main__':
    main()