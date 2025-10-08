import time
import torch
from torch import nn
import contextlib
from d2l.torch import accuracy, Accumulator

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.full_connection = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 79),
        )

    def forward(self, x):
        x = self.convolution(x)
        x = torch.flatten(x, 1)
        x = self.full_connection(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate_model(model, data_iter, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_model(para_model, para_train_iter, para_test_iter, num_epochs, lr, para_device):
    if not torch.cuda.is_available() and para_device.type == 'cuda':
        print("Warning: CUDA disabled, using CPU instead.")
        para_device = torch.device('cpu')

    para_model.apply(init_weights)
    print('training on', para_device)
    para_model.to(para_device)

    optimizer = torch.optim.SGD(
        para_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    loss = nn.CrossEntropyLoss(reduction='mean')

    if para_device.type == 'cuda':
        from torch.cuda.amp import GradScaler, autocast
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

        @contextlib.contextmanager
        def autocast():
            yield

    best_test_acc = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        para_model.train()

        num_batches = len(para_train_iter)

        for i, (X, y) in enumerate(para_train_iter):
            optimizer.zero_grad()
            X, y = X.to(para_device), y.to(para_device)

            with torch.amp.autocast(device_type=para_device.type):
                y_hat = para_model(X)
                l = loss(y_hat, y)

            if scaler is not None:
                scaler.scale(l).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(para_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm_(para_model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]

        scheduler.step()

        test_acc = evaluate_model(para_model, para_test_iter, para_device)
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"LR: {current_lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
              f"Test Acc: {test_acc:.3f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = para_model.state_dict().copy()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Duration: {training_time:.2f} seconds")
    print(f"Best Accuracy: {best_test_acc:.3f}")

    if best_model_state is not None:
        para_model.load_state_dict(best_model_state)
    return para_model