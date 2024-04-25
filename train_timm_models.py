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
import numpy as np
from tqdm.auto import tqdm
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from general import test_epoch, train_epoch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--model-names", nargs='+')
    parser.add_argument("--n-classes", type=int)
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


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs=10, device='cpu', experiment_id=None):
    model_name = model.__class__.__name__
    val_loss_history = []
    val_acc_history = []
    artifact = wandb.Artifact(f'{model_name}_weights', type='model')
    
    run = wandb.init(
        project="PTX",
        group=f'Ensemble_Training_{experiment_id}',
        name=model_name,
        job_type='train',
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "droupout": args.dropout_rate,
            "learning_rate": args.lr,
            "training_image_size": args.train_image_size,
        })
    
    for epoch in tqdm(range(epochs),
                      desc=f'Training {model_name}',
                      position=0,
                      leave=True):
        train_loss, train_accuracy, train_sensitivity, train_specificity = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_accuracy, val_sensitivity, val_specificity = test_epoch(model, val_dataloader, loss_fn, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)
        macro_avg_val_sensitivity = np.nanmean(val_sensitivity, axis=0)
        macro_avg_val_specificity = np.nanmean(val_specificity, axis=0)
        macro_avg_train_sensitivity = np.nanmean(train_sensitivity, axis=0)
        macro_avg_train_specificity = np.nanmean(train_specificity, axis=0)

        if val_accuracy == max(val_acc_history) and val_loss == min(val_loss_history):
            torch.save(model.state_dict(), f'weights/{model_name}_best.pt')            

        wandb.log({f'train_loss': train_loss,
                   f'train_acc': train_accuracy,
                   f'train_sensitivity': macro_avg_train_sensitivity,
                   f'train_specificity': macro_avg_train_specificity,
                   f'val_loss': val_loss,
                   f'val_acc': val_accuracy,
                   f'val_sensitivity': macro_avg_val_sensitivity,
                   f'val_specificity': macro_avg_val_specificity})
        
    torch.save(model.state_dict(), f'weights/{model_name}_last.pt')
    artifact.add_file(f'weights/{model_name}_last.pt')
    artifact.add_file(f'weights/{model_name}_best.pt')
    artifact.save()
    run.log_artifact(artifact)
    run.finish()


def train_all_models(args):
    
    experiment_id = wandb.util.generate_id()

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(args.train_image_size, args.train_image_size), antialias=True),
        v2.CenterCrop(size=(args.train_image_size, args.train_image_size)),
        v2.RandomEqualize(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2),
        v2.RandomRotation(degrees=10, interpolation='bilinear')
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
    
    for model in models:
        model.reset_classifier(args.n_classes, 'max')
    
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
              device=DEVICE,
              experiment_id=experiment_id)


if __name__ == "__main__":
    args = parse_args()
    train_all_models(args)