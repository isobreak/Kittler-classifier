import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from CNN.models import *


def train_cnn(dataset_path: str, model,
              save_csv_path: str, save_model_folder: str, model_name: str, save_plots_folder: str,
              learning_params: dict[int, Any, float]):
    """
    Trains model, saves results according to paths
    :param dataset_path: path to a dataset folder with two class folders
    :param model: model to be trained
    :param save_csv_path: path for saving/updating csv file with metrics
    :param save_model_folder: path for saving model
    :param model_name: name of the model to be used for savings (csv, model, plots)
    :param save_plots_folder: path for saving plots (old plots with same names will be rewritten)
    :param learning_params: dictionary with learning params {"epochs", "optimizer", "lr", "batch_size"}
    :return: trained model
    """
    # setting device
    device = ('cuda:0' if torch.cuda.is_available else 'cpu')
    print('Device:', device)

    print('Model name: ', model_name)

    # folders creation
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)
    if not os.path.exists(save_plots_folder):
        os.mkdir(save_plots_folder)

    # fitting model, gathering metrics
    model = model.to(device)
    model, metrics = fit_model(model, root=dataset_path, device=device, **learning_params,
                               print_train_metrics=False, print_test_metrics=True)
    # saving plots
    if save_plots_folder is not None:
        train_losses = metrics['train']['losses']
        train_accuracy = metrics['train']['accuracy']
        train_precision = metrics['train']['precision']
        train_recall = metrics['train']['recall']
        test_losses = metrics['test']['losses']
        test_accuracy = metrics['test']['accuracy']
        test_precision = metrics['test']['precision']
        test_recall = metrics['test']['recall']

        plt.plot(train_losses, label='Train losses')
        plt.plot(test_losses, label='Test losses')
        plt.legend()
        plt.savefig(os.path.join(save_plots_folder, model_name + '_losses.png'))
        plt.clf()

        plt.plot(train_accuracy, label='Train accuracy')
        plt.plot(test_accuracy, label='Test accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_plots_folder, model_name + '_accuracy.png'))
        plt.clf()

        plt.plot(train_precision, label='Train precision')
        plt.plot(test_precision, label='Test precision')
        plt.plot(train_recall, label='Train recall')
        plt.plot(test_recall, label='Test recall')
        plt.legend()
        plt.savefig(os.path.join(save_plots_folder, model_name + '_pre_rec.png'))
        plt.clf()

        print(f'Plots have been saved at {save_plots_folder}')

    # best f1-score search
    b_f = 0
    b_i = 0
    for i in range(learning_params['epochs']):
        a = metrics['test']['accuracy'][i]
        p = metrics['test']['precision'][i]
        r = metrics['test']['recall'][i]
        if p == 0 or r == 0:
            continue
        f = (2*p*r) / (p+r)
        if f > b_f and a > 0.5 and p > 0.5 and r > 0.5:
            b_f = f
            b_i = i

    a = "%.5f" % metrics['test']['accuracy'][b_i]
    p = "%.5f" % metrics['test']['precision'][b_i]
    r = "%.5f" % metrics['test']['recall'][b_i]
    print(f'a={a}, p={p}, r={r}, f(max)={b_f} - epoch({b_i + 1})\n')

    a = "%.0f" % (metrics['test']['accuracy'][b_i] * 100)
    p = "%.0f" % (metrics['test']['precision'][b_i] * 100)
    r = "%.0f" % (metrics['test']['recall'][b_i] * 100)
    b_f = "%.0f" % (b_f * 100)

    # updating existing info
    info = [a, p, r, b_f, b_i + 1]
    if save_csv_path is not None:
        if (os.path.exists(save_csv_path)):
            df = pd.read_csv(save_csv_path)
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
        else:
            df = pd.DataFrame(columns=['model_name', 'accuracy', 'precision', 'recall', 'f1-score', 'best_epoch'])

        df.loc[len(df.index)] = [model_name] + info
        df.to_csv(save_csv_path)
        print(f'csv has been saved at {save_csv_path}')

    # saving model
    if save_model_folder is not None:
        torch.save(model, os.path.join(save_model_folder, model_name + '.pt'))
        print(f'Model {model_name} has been saved at {save_model_folder}')

    return model


def fit_model(model, root: str, device: str = 'cpu', epochs: int = 25,
              print_train_metrics: bool = False, print_test_metrics: bool = False,
              optimizer=torch.optim.Adam, lr=0.001, batch_size=20) -> tuple[Any, dict[str, dict[str, int]]]:
    """
    Fits model and gathers information about metrics every epoch
    :param model: model to be trained
    :param root: path to dataset directory of given structure: class 1, class2
    :param device: 'cuda' or 'cpu'
    :param epochs: number of epochs
    :param optimizer: optimizer to be used
    :param lr: learning rate
    :param batch_size: batch size
    :param print_train_metrics: if it is necessary to print test metrics during training
    :param print_test_metrics: if it is necessary to print test metrics during training
    :return: trained model, dict with metrics [test, train][loss, accuracy, precision, recall]
    """

    # hyperparameters
    loss_function = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    common_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Resize((250, 250)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # preparing dataloaders: train, test
    print(f"Dataset's location: {root}")
    full_dataset = torchvision.datasets.ImageFolder(root, transform=common_transforms)
    classes = full_dataset.classes

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f'Train: {train_size}, test: {test_size}, epochs: {epochs}')

    # train info
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []

    # fitting
    for epoch in range(epochs):
        epoch += 1

        train_loss_sum = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train * 2 - 1
            X_train = X_train.to(device)
            y_train = y_train.to(device).to(torch.float32)

            y_pred = model(X_train)
            loss = loss_function(y_pred, y_train)
            train_loss_sum += loss.item()

            y_pred = y_pred.to('cpu')
            y_train = y_train.to('cpu')
            preds_train_b = torch.round(torch.nn.functional.sigmoid(y_pred))
            if b == 0:
                preds_train = np.array(preds_train_b.detach())
                ys_train = np.array(y_train.detach())
            else:
                preds_train = np.concatenate((preds_train, preds_train_b.detach()), axis=y_train.dim() - 1)
                ys_train = np.concatenate((ys_train, y_train.detach()), axis=y_train.dim() - 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_train = accuracy_score(ys_train, preds_train)
        pre_train = precision_score(ys_train, preds_train, zero_division=0.0)
        rec_train = recall_score(ys_train, preds_train)
        train_losses.append(train_loss_sum / train_size)
        train_accuracy.append(acc_train)
        train_precision.append(pre_train)
        train_recall.append(rec_train)

        if print_train_metrics is True:
            print(f'\nEpoch: {epoch}')
            print(f'TRAIN: loss = {"%.5f" % loss}, acc = {"%.5f" % acc_train}, pre = {"%.5f" % pre_train}, '
                  f'rec = {"%.5f" % rec_train}, f1 = {"%.5f" % (2*rec_train*pre_train/(rec_train+pre_train))}')
            print('Confusion matrix for train:\n', confusion_matrix(ys_train, preds_train))

        # estimating
        with torch.no_grad():
            test_loss_sum = 0
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test * 2 - 1
                X_test = X_test.to(device)
                y_test = y_test.to(device).to(torch.float32)

                y_val = model(X_test)
                loss = loss_function(y_val, y_test)
                test_loss_sum += loss.item()

                y_val = y_val.to('cpu')
                y_test = y_test.to('cpu')
                preds_test_b = torch.round(torch.nn.functional.sigmoid(y_val))
                if b == 0:
                    preds_test = np.array(preds_test_b.detach())
                    ys_test = np.array(y_test.detach())
                else:
                    preds_test = np.concatenate((preds_test, preds_test_b.detach()), axis=y_test.dim() - 1)
                    ys_test = np.concatenate((ys_test, y_test.detach()), axis=y_test.dim() - 1)

            test_losses.append(test_loss_sum / test_size)
            acc_test = accuracy_score(ys_test, preds_test)
            pre_test = precision_score(ys_test, preds_test, zero_division=0.0)
            rec_test = recall_score(ys_test, preds_test)

            f1 = f1_score(ys_test, preds_test)

            test_accuracy.append(acc_test)
            test_precision.append(pre_test)
            test_recall.append(rec_test)
            if print_test_metrics is True:
                if print_train_metrics is not True:
                    print(f'\nEpoch: {epoch}')
                print(f'Test metrics for {classes[1]}\nacc = {"%.5f" % acc_test}, pre = {"%.5f" % pre_test}, '
                      f'rec = {"%.5f" % rec_test}, f1: {"%.5f" % f1}')
                print('Confusion matrix:\n', confusion_matrix(ys_test, preds_test))

    return model, dict(train={'losses': train_losses,
                              'accuracy': train_accuracy,
                              'precision': train_precision,
                              'recall': train_recall},
                       test={'losses': test_losses,
                             'accuracy': test_accuracy,
                             'precision': test_precision,
                             'recall': test_recall})