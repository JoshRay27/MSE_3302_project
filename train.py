from torch.utils.data import DataLoader, random_split
import torch
from dataset import PreprocessedImageDataset
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from models.SVM import SVMClassifier
from training import train, evaluate

DATA_DIR = "data_0_1/"
BATCH_SIZE = 32
NUM_CLASSES = 2

def main():
    # load full dataset
    dataset = PreprocessedImageDataset(DATA_DIR)

    # compute split sizes
    total = len(dataset)
    print(f"Length of dataset: {total}")

    train_size = int(0.9 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size

    #Perform Split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SimpleCNN(num_classes=NUM_CLASSES)
    train(model, train_loader, epochs=10, lr=1e-3)
    torch.save(model.state_dict(), "simple_cnn_model.pth")
    print("model saved")
    print("Model_CNN Evaluation")
    print(evaluate(model, val_loader))

    #model_SVM = SVMClassifier(input_dim=input_dim, num_classes=10)
    #train(model_SVM, train_loader, epochs=5, lr=1e-3)
    
    '''model = ASLNet(num_classes= NUM_CLASSES)
    train(model, train_loader, epochs=10, lr=1e-3)
    torch.save(model.state_dict(), "cnn_model.pth")
    print("model saved")
    print("Model_ASL Evaluation")
    print(evaluate(model, val_loader))'''

if __name__ == "__main__":
    main()
