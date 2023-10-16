"""
@author: Robert Szabo
email: robert.szabo@irob.uni-obuda.hu

Trains a number of timm models on a given dataset, and saves the experiment in Weights & Biases. Login to Weights & Biases is a prerequisite.
Uses the same training parameters for all models.

Parameters:
    --data-folder: path to the dataset root folder. The dataset should be split into "train" and "validation" folders.
    --model-names: list of model names to train. The models should be available in timm.
    --n-classes: number of classes in the dataset.
    --batch-size: batch size for training.
    --epochs: number of epochs to train.
    --lr: learning rate.
    --dropout-rate: dropout rate for the models.
    --train-image-size: size of the images to train on. The images will be resized to this size. Should be quadratic.
    --print-progress: Setting it to True will print accuracy and loss after each epoch. Defaulted to false, as progress is visible in Weights & Biases.
"""

import torch
import timm
import wandb
from tqdm.auto import tqdm
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--model-names", nargs='+')
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--train-image-size", type=int, default=256)
    parser.add_argument("--print-progress", type=bool, default=False)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= size
    correct /= size
    return train_loss, correct


def test_epoch(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    return test_loss, correct


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs=10, device='cpu'):
    model_name = model.__class__.__name__
    val_loss_history = []
    val_acc_history = []
    artifact = wandb.Artifact(f'{model_name}_weights', type='model')
    
    run = wandb.init(
        project="PTX",
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "droupout": args.dropout_rate,
            "learning_rate": args.lr,
            "training_image_size": args.train_image_size
        })
    
    for epoch in tqdm(range(epochs),
                      desc=f'Training {model_name}',
                      position=0,
                      leave=True):
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = test_epoch(model, val_dataloader, loss_fn, device)
        if args.print_progress:
            print(f'Epoch #{epoch+1}: train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_acc == max(val_acc_history) and val_loss == min(val_loss_history):
            torch.save(model.state_dict(), f'{model_name}_best.pt')            

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        
    torch.save(model.state_dict(), f'{model_name}_last.pt')
    artifact.add_file(f'{model_name}_last.pt')
    artifact.add_file(f'{model_name}_best.pt')
    artifact.save()
    run.log_artifact(artifact)
    run.finish()


def train_all_models(args):
    transforms = Compose([
        ToTensor(),
        Resize(size=(args.train_image_size, args.train_image_size), antialias=True),
        CenterCrop(size=(args.train_image_size, args.train_image_size))
    ])

    train_dataset = ImageFolder(root=f'{args.data_folder}/train', transform=transforms)
    val_dataset = ImageFolder(root=f'{args.data_folder}/validation', transform=transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)
    
    models = [timm.create_model(model_name,
                                pretrained=True,
                                num_classes=args.n_classes,
                                drop_rate=args.dropout_rate) for model_name in args.model_names]
    
    loss_fn = torch.nn.CrossEntropyLoss() if args.n_classes > 2 else torch.nn.BCEWithLogitsLoss()
    
    print(f'Starting training for the following models:\n{args.model_names}')
    for model in models:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model=model.to(DEVICE),
              train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=args.epochs,
              device=DEVICE)


if __name__ == "__main__":
    args = parse_args()
    train_all_models(args)