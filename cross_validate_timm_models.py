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
"""

import torch
import timm
import wandb
import sys
from tqdm.auto import tqdm
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torch.utils.data import DataLoader
from general import train_epoch, test_epoch, get_metrics_from_confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--model-names", nargs='+')
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--train-image-size", type=int, default=256)
    try:
        return parser.parse_args()
    except SystemExit as err:
        sys.traceback.print_exc()
        sys.exit(err.code)


def cross_validate_model(model_name, dataset, k_folds, loss_fn, epochs=10, device='cpu'):
    total_size = len(dataset)
    split_size = total_size // k_folds
    experiment_id = wandb.util.generate_id()
    
    for fold in tqdm(range(k_folds),
                     desc=f"Performing cross validation for {model_name}"):

        run = wandb.init(
            project="PTX",
            group=f'cross validation {model_name}_{experiment_id}',
            name=f'fold_{fold+1}',
            job_type='eval',
            config={
                "task": "cross validation",
                "model": model_name,
                "epochs": epochs,
                "validation folds": k_folds,
                "batch_size": args.batch_size,
                "droupout": args.dropout_rate,
                "learning_rate": args.lr,
                "training_image_size": args.train_image_size,
            })

        model = timm.create_model(model_name,
                                  pretrained=True,
                                  num_classes=args.n_classes,
                                  drop_rate=args.dropout_rate)        
        model.reset_classifier(args.n_classes, 'max')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [total_size - split_size, split_size])

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
        
        for epoch in tqdm(range(epochs),
                          desc=f'Training {model_name}',
                          position=0,
                          leave=True):
            train_loss, train_accuracy, train_sensitivity, train_specificity = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
            val_loss, val_accuracy, val_sensitivity, val_specificity = test_epoch(model, val_dataloader, loss_fn, device)
            macro_avg_val_sensitivity = sum(val_sensitivity) / len(val_sensitivity)
            macro_avg_val_specificity = sum(val_specificity) / len(val_specificity)
            macro_avg_train_sensitivity = sum(train_sensitivity) / len(train_sensitivity)
            macro_avg_train_specificity = sum(train_specificity) / len(train_specificity)

            wandb.log({f'train_loss': train_loss,
                       f'train_acc': train_accuracy,
                       f'train_sensitivity': macro_avg_train_sensitivity,
                       f'train_specificity': macro_avg_train_specificity,
                       f'val_loss': val_loss,
                       f'val_acc': val_accuracy,
                       f'val_sensitivity': macro_avg_val_sensitivity,
                       f'val_specificity': macro_avg_val_specificity})
        
        run.finish()


def cross_validate_all_models(args):
    transforms = Compose([
        ToTensor(),
        Resize(size=(args.train_image_size, args.train_image_size), antialias=True),
        CenterCrop(size=(args.train_image_size, args.train_image_size))
    ])

    train_dataset = ImageFolder(root=f'{args.data_folder}/train', transform=transforms)
    
    loss_fn = torch.nn.CrossEntropyLoss() if args.n_classes > 2 else torch.nn.BCEWithLogitsLoss()
    
    print(f'Starting cross validation for the following models:\n{args.model_names}')
    for model_name in args.model_names:
        cross_validate_model(model_name=model_name,
                             dataset=train_dataset,
                             k_folds=args.k_folds,
                             loss_fn=loss_fn,
                             epochs=args.epochs,
                             device=DEVICE)


if __name__ == "__main__":
    args = parse_args()
    cross_validate_all_models(args)