import torch
import pandas as pd
import seaborn as sn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss, train_acc = 0, 0
    train_sensitivity, train_specificity = [], []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        pred = output.softmax(1).argmax(1)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        acc, sensitivity, specificity = get_metrics_from_confusion_matrix(y, pred) #TODO: fix n_classes parameter for subsets
        train_sensitivity.append(sensitivity)
        train_specificity.append(specificity)
        train_acc += acc

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    #concat_sens = np.stack(test_sensitivity, axis=0)
    train_sensitivity = np.nanmean(np.stack(train_sensitivity, axis=0), axis=0)
    train_specificity = np.nanmean(np.stack(train_specificity, axis=0), axis=0)
    
    return train_loss, train_acc, train_sensitivity, train_specificity


def pad_array_with_nan(array, length):
    if len(array) < length:
        array = np.pad(array, (0, length - len(array)), 'constant', constant_values=np.nan)
    return array


def test_epoch(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, test_acc = 0, 0
    test_sensitivity, test_specificity = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = output.softmax(1).argmax(1)

            test_loss += loss_fn(output, y).item()
            acc, sensitivity, specificity = get_metrics_from_confusion_matrix(y, pred)  #TODO: fix n_classes parameter for subsets
            test_sensitivity.append(sensitivity)
            test_specificity.append(specificity)
            test_acc += acc
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_sensitivity = np.nanmean(np.stack(test_sensitivity, axis=0), axis=0)
    test_specificity = np.nanmean(np.stack(test_specificity, axis=0), axis=0)

    return test_loss, test_acc, test_sensitivity, test_specificity


def validate(model, val_dataloader, device='cpu'):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, acc, sens, spec = test_epoch(model, val_dataloader, loss_fn, device)
    #print(f'loss: {loss:.4f}, acc: {acc:.4f}, sens: {sens}, spec: {spec}')
    print(f'loss: {loss:.4f}, acc: {acc:.4f}, sens: {np.nanmean(sens):.4f}, spec: {np.nanmean(spec):.4f}')
    return loss, acc, sens, spec


def display_confusion_matrix(true_labels, predicted_labels, class_labels):
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    if cf_matrix.shape[0] < len(class_labels):
        cf_matrix = np.pad(cf_matrix, (0, len(class_labels) - cf_matrix.shape[0]), 'constant')
    
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                         index = [i for i in class_labels],
                         columns = [i for i in class_labels])
    plt.figure(figsize = (7,7))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')


#TODO: rework with monai confmat metrics
#TODO: remove n_classes default once I can reasonably pass it over
#TODO: check if classification report from sklearn works
def get_metrics_from_confusion_matrix(true_labels, predicted_labels, n_classes=4):
    cf_matrix = confusion_matrix(true_labels.cpu(), predicted_labels.cpu())
    padded_cf_matrix = np.zeros((n_classes, n_classes))
    padded_cf_matrix[:cf_matrix.shape[0], :cf_matrix.shape[1]] = cf_matrix
    
    sensitivity_per_class = np.zeros(n_classes)
    specificity_per_class = np.zeros(n_classes)
    accuracy = np.sum(np.diag(padded_cf_matrix)) / np.sum(padded_cf_matrix)

    for class_id in range(n_classes):
        sensitivity_per_class[class_id] = padded_cf_matrix[class_id, class_id] / np.sum(padded_cf_matrix[class_id, :])
        specificity_per_class[class_id] = padded_cf_matrix[class_id, class_id] / np.sum(padded_cf_matrix[:, class_id])

    return accuracy, sensitivity_per_class, specificity_per_class


def gradcam_overlay(image, gradcam, colormap='jet', opacity=0.2):
    plt.imshow(image, cmap='gray')
    plt.imshow(gradcam, cmap=colormap, alpha=opacity)