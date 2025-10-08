import torch
from torch import multiprocessing
from torch.utils.data import DataLoader
from data_loader import ImageDataset
from alexnet import AlexNet, train_model

def main():
    batch_size = 256
    learning_rate = 0.1
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dataset = ImageDataset('datasets')

    train_iter = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8, prefetch_factor=2, persistent_workers=True)
    test_iter = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=8, prefetch_factor=2, persistent_workers=True)

    model = AlexNet().to(device)
    trained_model = train_model(model, train_iter, test_iter, num_epochs, learning_rate, device)

    torch.save(trained_model.state_dict(), "model.pth")
    print("Saving model to model.pth!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()