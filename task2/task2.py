import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


# ---------------------------
# MyExtremeLearningMachine
# ---------------------------
class MyExtremeLearningMachine(nn.Module):
    def __init__(self, num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1):
        """
        ELM with one fixed convolutional layer and one trainable FC layer.
        """
        super(MyExtremeLearningMachine, self).__init__()
        self.fixed_conv = nn.Conv2d(num_channels, hidden_maps, kernel_size, padding=1)
        self.initialise_fixed_layers(std)

        # Freeze the fixed convolutional layer
        for param in self.fixed_conv.parameters():
            param.requires_grad = False

        # Trainable fully-connected layer
        self.fc = nn.Linear(hidden_maps * 32 * 32, num_classes)

    def initialise_fixed_layers(self, std):
        nn.init.normal_(self.fixed_conv.weight, mean=0.0, std=std)
        if self.fixed_conv.bias is not None:
            nn.init.constant_(self.fixed_conv.bias, 0.0)

    def forward(self, x):
        x = self.fixed_conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ---------------------------
# MyMixUp for Data Augmentation
# ---------------------------
class MyMixUp:
    def __init__(self, alpha=0.2, seed=42):
        """
        Mixup augmentation.
        """
        self.alpha = alpha
        np.random.seed(seed)

    def __call__(self, images, labels):
        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices
        index = torch.randperm(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]

        # Convert labels to one-hot
        num_classes = 10
        labels_onehot = torch.zeros(batch_size, num_classes, device=images.device)
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)

        labels_shuffled = torch.zeros(batch_size, num_classes, device=images.device)
        labels_shuffled.scatter_(1, labels[index].view(-1, 1), 1)

        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled
        return mixed_images, mixed_labels


# ---------------------------
# MyEnsembleELM for Model Ensemble
# ---------------------------
class MyEnsembleELM:
    def __init__(self, num_models=3, seed=42, **elm_kwargs):
        """
        Ensemble of ELM models.
        """
        if num_models < 1 or num_models > 10:
            raise ValueError("num_models must be between 1 and 10.")
        self.num_models = num_models
        self.models = []
        for i in range(num_models):
            torch.manual_seed(seed + i)
            model = MyExtremeLearningMachine(**elm_kwargs)
            self.models.append(model)

    def predict(self, x):
        preds = [model(x) for model in self.models]
        return torch.mean(torch.stack(preds), dim=0)


# ---------------------------
# Helper Functions
# ---------------------------
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def compute_f1(outputs, labels, num_classes=10):
    """
    Compute macro F1-score for multi-class classification.
    """
    _, predicted = torch.max(outputs, 1)

    f1_scores = []
    for cls in range(num_classes):
        true_positive = ((predicted == cls) & (labels == cls)).sum().item()
        false_positive = ((predicted == cls) & (labels != cls)).sum().item()
        false_negative = ((predicted != cls) & (labels == cls)).sum().item()

        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes


def create_montage(images, labels, predictions, classes, save_path="result.png"):
    """
    Create and save a montage of images with ground-truth and predicted labels.
    Assumes images are normalized (mean=0.5, std=0.5).
    """
    # Denormalize
    images = images * 0.5 + 0.5
    images_np = images.cpu().numpy()
    montage_list = []
    try:
        font = ImageFont.truetype("arial.ttf", 8)
    except IOError:
        print("TTF font not found; using default font instead.")
        font = ImageFont.load_default()

    for i in range(images_np.shape[0]):
        img = np.transpose(images_np[i], (1, 2, 0)) * 255
        img = img.astype(np.uint8)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        text = f"GT: {classes[labels[i]]}\nP: {classes[predictions[i]]}"
        draw.text((5, 5), text, fill=(255, 0, 0), font=font)
        montage_list.append(pil_img)

    grid_size = int(np.sqrt(len(montage_list)))
    width, height = montage_list[0].size
    montage_img = Image.new('RGB', (width * grid_size, height * grid_size))

    for idx, img in enumerate(montage_list):
        row = idx // grid_size
        col = idx % grid_size
        montage_img.paste(img, (col * width, row * height))

    montage_img.save(save_path)
    print(f"Montage saved as {save_path}")


def create_mixup_montage(images, save_path="mixup.png", grid_size=4):
    """
    Create and save a montage of mixup augmented images.
    """
    images = images * 0.5 + 0.5
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


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate a single model (non-ensemble) on a given dataloader.
    Returns: avg_loss, accuracy (%), f1
    """
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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


def evaluate_ensemble(model, dataloader, device):
    """
    Evaluate an ensemble model on a given dataloader.
    Returns: avg_loss, accuracy (%), f1
    """
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


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Baseline ELM
    # ---------------------------
    print("Training baseline ELM (without regularisation)...")
    model_baseline = MyExtremeLearningMachine(num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1)
    optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=0.005, momentum=0.9)
    epochs_baseline = 20

    # We'll store all epochs' metrics here
    history_baseline = []
    baseline_ckpt_path = "baseline_elm_checkpoint.pth"
    start_epoch_baseline = 0

    # If checkpoint exists, load it (including history if available)
    if os.path.exists(baseline_ckpt_path):
        checkpoint = torch.load(baseline_ckpt_path)
        model_baseline.load_state_dict(checkpoint['model_state_dict'])
        optimizer_baseline.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch_baseline = checkpoint['epoch']

        # If 'history' was saved, load and print it
        if 'history' in checkpoint:
            history_baseline = checkpoint['history']
            print("==== Loaded Baseline ELM training history from checkpoint ====")
            for entry in history_baseline:
                print(f"Epoch {entry['epoch']}: "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry['train_f1']:.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print(f"Loaded Baseline ELM checkpoint from epoch {start_epoch_baseline}")

    model_baseline.to(device)
    for epoch in range(start_epoch_baseline, epochs_baseline):
        # Training loop
        model_baseline.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_baseline.zero_grad()
            outputs = model_baseline(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_baseline.step()

        # Evaluate on train
        train_loss, train_acc, train_f1 = evaluate_model(model_baseline, trainloader, criterion, device)
        # Evaluate on test
        test_loss, test_acc, test_f1 = evaluate_model(model_baseline, testloader, criterion, device)

        print(f"[Baseline ELM] Epoch {epoch + 1}/{epochs_baseline} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

        # Store metrics in history
        history_baseline.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_baseline.state_dict(),
            'optimizer_state_dict': optimizer_baseline.state_dict(),
            'history': history_baseline
        }, baseline_ckpt_path)
        print(f"Checkpoint saved at {baseline_ckpt_path}")

    # ---------------------------
    # MixUp ELM
    # ---------------------------
    print("\nGenerating MixUp augmentation montage (mixup.png)...")
    mixup = MyMixUp(alpha=0.2, seed=42)
    mixup_batch = next(iter(trainloader))
    images, labels = mixup_batch
    images, labels = images.to(device), labels.to(device)
    mixed_images, _ = mixup(images, labels)
    create_mixup_montage(mixed_images, save_path="mixup.png", grid_size=4)

    print("\nTraining ELM with MixUp regularisation...")
    model_mixup = MyExtremeLearningMachine(num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1)
    optimizer_mixup = optim.SGD(model_mixup.parameters(), lr=0.005, momentum=0.9)
    epochs_mixup = 20

    history_mixup = []
    mixup_ckpt_path = "mixup_elm_checkpoint.pth"
    start_epoch_mixup = 0

    # Load checkpoint if exists
    if os.path.exists(mixup_ckpt_path):
        checkpoint = torch.load(mixup_ckpt_path)
        model_mixup.load_state_dict(checkpoint['model_state_dict'])
        optimizer_mixup.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch_mixup = checkpoint['epoch']

        if 'history' in checkpoint:
            history_mixup = checkpoint['history']
            print("==== Loaded MixUp ELM training history from checkpoint ====")
            for entry in history_mixup:
                print(f"Epoch {entry['epoch']}: "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry['train_f1']:.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print(f"Loaded MixUp ELM checkpoint from epoch {start_epoch_mixup}")

    model_mixup.to(device)
    for epoch in range(start_epoch_mixup, epochs_mixup):
        # Training
        model_mixup.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, mixed_labels = mixup(inputs, labels)
            hard_labels = torch.argmax(mixed_labels, dim=1)
            optimizer_mixup.zero_grad()
            outputs = model_mixup(inputs)
            loss = criterion(outputs, hard_labels)
            loss.backward()
            optimizer_mixup.step()

        # Evaluate
        train_loss, train_acc, train_f1 = evaluate_model(model_mixup, trainloader, criterion, device)
        test_loss, test_acc, test_f1 = evaluate_model(model_mixup, testloader, criterion, device)

        print(f"[MixUp ELM] Epoch {epoch + 1}/{epochs_mixup} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

        history_mixup.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_mixup.state_dict(),
            'optimizer_state_dict': optimizer_mixup.state_dict(),
            'history': history_mixup
        }, mixup_ckpt_path)
        print(f"Checkpoint saved at {mixup_ckpt_path}")

    # ---------------------------
    # Ensemble ELM
    # ---------------------------
    print("\nTraining Ensemble ELM...")
    ensemble_model = MyEnsembleELM(num_models=3, num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1)
    ensemble_epochs = 20
    ensemble_optimizers = [optim.SGD(m.parameters(), lr=0.001, momentum=0.9) for m in ensemble_model.models]

    history_ensemble = []
    ensemble_ckpt_path = "ensemble_elm_checkpoint.pth"
    start_epoch_ensemble = 0

    # Load checkpoint if exists
    if os.path.exists(ensemble_ckpt_path):
        checkpoint = torch.load(ensemble_ckpt_path)
        start_epoch_ensemble = checkpoint['epoch']
        for i, sub_model in enumerate(ensemble_model.models):
            sub_model.load_state_dict(checkpoint[f'model_{i}_state_dict'])
            ensemble_optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
        if 'history' in checkpoint:
            history_ensemble = checkpoint['history']
            print("==== Loaded Ensemble ELM training history from checkpoint ====")
            for entry in history_ensemble:
                print(f"Epoch {entry['epoch']}: "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry.get('train_f1',0):.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print(f"Loaded Ensemble ELM checkpoint from epoch {start_epoch_ensemble}")

    for epoch in range(start_epoch_ensemble, ensemble_epochs):
        # Train each model in the ensemble
        for idx, model in enumerate(ensemble_model.models):
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                ensemble_optimizers[idx].zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                ensemble_optimizers[idx].step()

        # Evaluate on train
        train_loss, train_acc, train_f1 = evaluate_ensemble(ensemble_model, trainloader, device)
        # Evaluate on test
        test_loss, test_acc, test_f1 = evaluate_ensemble(ensemble_model, testloader, device)

        print(f"[Ensemble ELM] Epoch {epoch + 1}/{ensemble_epochs} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

        history_ensemble.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

        ckpt_dict = {'epoch': epoch + 1, 'history': history_ensemble}
        for i, sub_model in enumerate(ensemble_model.models):
            ckpt_dict[f'model_{i}_state_dict'] = sub_model.state_dict()
            ckpt_dict[f'optimizer_{i}_state_dict'] = ensemble_optimizers[i].state_dict()
        torch.save(ckpt_dict, ensemble_ckpt_path)
        print(f"Checkpoint saved at {ensemble_ckpt_path}")

    # ---------------------------
    # Ensemble ELM with MixUp
    # ---------------------------
    print("\nTraining Ensemble ELM with MixUp (Combined Regularisation) with epoch evaluation...")
    ensemble_mixup_models = [
        MyExtremeLearningMachine(num_channels=3, hidden_maps=16, num_classes=10, kernel_size=3, std=0.1).to(device)
        for _ in range(3)
    ]
    ensemble_mixup_epochs = 20
    ensemble_mixup_optimizers = [
        optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
        for m in ensemble_mixup_models
    ]

    history_ensemble_mixup = []
    ensemble_mixup_ckpt_path = "ensemble_mixup_elm_checkpoint.pth"
    start_epoch_ensemble_mixup = 0

    if os.path.exists(ensemble_mixup_ckpt_path):
        checkpoint = torch.load(ensemble_mixup_ckpt_path)
        start_epoch_ensemble_mixup = checkpoint['epoch']
        for i, sub_model in enumerate(ensemble_mixup_models):
            sub_model.load_state_dict(checkpoint[f'model_{i}_state_dict'])
            ensemble_mixup_optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
        if 'history' in checkpoint:
            history_ensemble_mixup = checkpoint['history']
            print("==== Loaded Ensemble MixUp ELM training history from checkpoint ====")
            for entry in history_ensemble_mixup:
                print(f"Epoch {entry['epoch']}: "
                      f"Train Loss={entry['train_loss']:.4f}, Train Acc={entry['train_acc']:.2f}%, Train F1={entry.get('train_f1',0):.4f}, "
                      f"Test Loss={entry['test_loss']:.4f}, Test Acc={entry['test_acc']:.2f}%, Test F1={entry['test_f1']:.4f}")
        print(f"Loaded Ensemble MixUp ELM checkpoint from epoch {start_epoch_ensemble_mixup}")

    class EnsembleMixUpELM:
        def __init__(self, models):
            self.models = models

        def predict(self, x):
            preds = [m(x) for m in self.models]
            return torch.mean(torch.stack(preds), dim=0)

    for epoch in range(start_epoch_ensemble_mixup, ensemble_mixup_epochs):
        for idx, model in enumerate(ensemble_mixup_models):
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, mixed_labels = mixup(inputs, labels)
                hard_labels = torch.argmax(mixed_labels, dim=1)
                ensemble_mixup_optimizers[idx].zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, hard_labels)
                loss.backward()
                ensemble_mixup_optimizers[idx].step()

        # Evaluate
        ensemble_mixup = EnsembleMixUpELM(ensemble_mixup_models)
        train_loss, train_acc, train_f1 = evaluate_ensemble(ensemble_mixup, trainloader, device)
        test_loss, test_acc, test_f1 = evaluate_ensemble(ensemble_mixup, testloader, device)

        print(f"[Ensemble MixUp ELM] Epoch {epoch + 1}/{ensemble_mixup_epochs} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

        history_ensemble_mixup.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

        ckpt_dict = {'epoch': epoch + 1, 'history': history_ensemble_mixup}
        for i, sub_model in enumerate(ensemble_mixup_models):
            ckpt_dict[f'model_{i}_state_dict'] = sub_model.state_dict()
            ckpt_dict[f'optimizer_{i}_state_dict'] = ensemble_mixup_optimizers[i].state_dict()
        torch.save(ckpt_dict, ensemble_mixup_ckpt_path)
        print(f"Checkpoint saved at {ensemble_mixup_ckpt_path}")

    # ---------------------------
    # Explain "Random Guess"
    # ---------------------------
    print("\nRandom Guess Explanation:")
    print("In multiclass classification, a random guess assigns each class equal probability (1/K for K classes).")
    print("For example, in a 10-class problem, the expected accuracy is around 10%.")
    print("To test this, generate random predictions uniformly and compare the measured accuracy against this baseline.")

    # ---------------------------
    # Final Reporting
    # ---------------------------
    print("\nMetrics Used:")
    print("Accuracy quantifies the proportion of correct predictions, providing a clear performance measure.")
    print("F1 score balances precision and recall, making it robust when class distributions are imbalanced.")

    # ---------------------------
    # Visualization
    # ---------------------------
    model_baseline.eval()
    test_iter = iter(testloader)
    images, labels = next(test_iter)
    images, labels = images.to(device), labels.to(device)
    outputs = model_baseline(images)
    _, predicted = torch.max(outputs, 1)
    num_images = min(36, images.size(0))
    create_montage(images[:num_images], labels[:num_images].cpu().tolist(),
                   predicted[:num_images].cpu().tolist(), classes, save_path="result.png")
    print("Montage saved as result.png.")

def print_checkpoint_history(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file '{ckpt_path}' does not exist.")
        return
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f"\nHistory from {ckpt_path}:")
    if 'history' not in checkpoint:
        print("No history field found in this checkpoint.")
        return
    history = checkpoint['history']
    for entry in history:
        print(f"Epoch {entry['epoch']}: "
              f"Train Loss: {entry['train_loss']:.4f}, "
              f"Train Acc: {entry['train_acc']:.2f}%, "
              f"Train F1: {entry['train_f1']:.4f}, "
              f"Test Loss: {entry['test_loss']:.4f}, "
              f"Test Acc: {entry['test_acc']:.2f}%, "
              f"Test F1: {entry['test_f1']:.4f}")

if __name__ == '__main__':
    main()
