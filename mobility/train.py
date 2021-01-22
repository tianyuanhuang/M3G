from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score
torch.cuda.is_available()

emb_input = torch.load(doc)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'mob_dist_chi'
embedding_dim = 200
num_places = 1294
num_contexts = 1294
threshold = 0.5
return_best = True
if_early_stop = True
input_size = 299
learning_rate = [0.001]
weight_decay = [0.0]
batch_size = 512
num_epochs = 51
lr_decay_rate = 0.7
lr_decay_epochs = 6
early_stop_epochs = 6
save_epochs = 1
# doc='data/docbar.tar'
margin = 2


class PlaceSkipGrammargin(nn.Module):
    def __init__(self, num_places=77744, num_contexts=756, embedding_dim=100):
        super(PlaceSkipGrammargin, self).__init__()
        self.place_embedding = nn.Embedding(num_places, embedding_dim)
        self.context_embedding = nn.Embedding(num_contexts, embedding_dim)

    def forward(self, place_indices, context_indices):
        z1 = self.place_embedding(place_indices)
        # N x embedding_dim
        z2 = self.context_embedding(context_indices)
        score = torch.nn.functional.pairwise_distance(
            z1, z2, p=2.0)               # N
        return score


class PlacePairDataset(Dataset):
    def __init__(self, path_list, weight, id_list, num):
        self.path_list = path_list
        self.weight = weight
        self.id_list = id_list
        self.num = num

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        pos_idx, place_idx = self.path_list[idx]
        media1 = copy.deepcopy(self.weight)
        media2 = np.zeros(self.num)
        media2[self.id_list[place_idx]] = self.weight[self.id_list[place_idx]]
        media1 = media1-media2
        media1 = np.power(media1, 0.5)
        media1 = media1/np.sum(media1)
        neg_idx = np.random.choice(np.arange(self.num), size=1, p=media1)[0]
        sample = [torch.tensor(place_idx, dtype=torch.long), torch.tensor(
            pos_idx, dtype=torch.long), torch.tensor(neg_idx, dtype=torch.long)]
        return sample


def metrics(stats):
    """
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    accuracy = (stats['T'] + 0.00001) * 1.0 / \
        (stats['T'] + stats['F'] + 0.00001)
    print(accuracy)
    return accuracy


def train_embedding(model, model_name, dataloaders, criterion, optimizer, metrics, num_epochs, threshold=0.5,
                    verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                    save_dir=None, save_epochs=1):
    since = time.time()
    training_log = dict()
    training_log['loss_history'] = []
    training_log['metric_value_history'] = []
    training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = 0.
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        running_loss = 0.0
        stats = {'T': 0, 'F': 0}
        # Iterate over data.
        for places, pos_index, neg_index in tqdm(dataloaders, ascii=True):
            places = places.to(device)
            pos_index = pos_index.to(device)
            neg_index = neg_index.to(device)
            labels = torch.ones(places.size(0))
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs1 = model(places, pos_index)
                outputs2 = model(places, neg_index)
                loss = criterion(outputs2, outputs1, target=labels)
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * places.size(0)
            stats['T'] += torch.sum(outputs1 < outputs2).cpu().item()
            stats['F'] += torch.sum(outputs1 > outputs2).cpu().item()
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_metric_value = metrics(stats)
        if verbose:
            print('Loss: {:.4f} Metrics: {:.4f}'.format(
                epoch_loss, epoch_metric_value))
        training_log['current_epoch'] = epoch
        training_log['metric_value_history'].append(epoch_metric_value)
        training_log['loss_history'].append(epoch_loss)
        if epoch_metric_value > best_metric_value:
            best_metric_value = epoch_metric_value
            best_model_wts = copy.deepcopy(model.state_dict())
            best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
            best_log = copy.deepcopy(training_log)
            nodecrease = 0
        else:
            nodecrease += 1
        if scheduler != None:
            scheduler.step()
        if nodecrease >= early_stop_epochs:
            early_stop = True
        if save_dir and epoch % save_epochs == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint, os.path.join(save_dir, model_name +
                                                '_' + str(training_log['current_epoch']) + '.tar'))
        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best metric value: {:4f}'.format(best_metric_value))

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }

    if save_dir:
        torch.save(checkpoint,
                   os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))
    return model, training_log, best_metric_value


if __name__ == '__main__':
    with open(test_pair, 'rb') as f:
        pair = pickle.load(f)
    with open(test_context, 'rb') as f:
        context = pickle.load(f)
    with open(test_count, 'rb') as f:
        count = pickle.load(f)
    print('amount of place: ' + str(len(context)))
    print('amount of pair: ' + str(len(pair)))
    datasets1 = PlacePairDataset(pair, count, context, count.shape[0])
    dataloaders_dict = DataLoader(
        datasets1, batch_size=batch_size, shuffle=True, num_workers=4)
    best_metric = 0
    best_lr = -1
    best_wr = -1
    for i in learning_rate:
        for j in weight_decay:
            model = PlaceSkipGrammargin(
                num_places=num_places, num_contexts=num_contexts, embedding_dim=embedding_dim)
            media_dir = os.path.join(ckpt_save_dir, "lr%f_wr%f" % (i, j))
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)
            model = model.to(device)
            if doc:
                checkpoint = torch.load(doc)
                model.load_state_dict(checkpoint, strict=False)
            optimizer = optim.Adam(model.parameters(), lr=i, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=j, amsgrad=True)
            loss_fn = torch.nn.MarginRankingLoss(
                reduction="mean", margin=margin)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)
            _, training_log, best_value = train_embedding(model, model_name=model_name, dataloaders=dataloaders_dict, criterion=loss_fn,
                                                          optimizer=optimizer, metrics=metrics, num_epochs=num_epochs, threshold=threshold,
                                                          verbose=True, return_best=return_best,
                                                          if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs, scheduler=scheduler,
                                                          save_dir=media_dir, save_epochs=save_epochs)
            media_dir1 = os.path.join(media_dir, "training_log.txt")
            with open(media_dir1, "w") as file:
                for k in range(len(training_log["metric_value_history"])):
                    file.write("epoch:"+str(k)+"\n")
                    file.write("metric_value_history:" +
                               str(training_log["metric_value_history"][k])+"\n")
                    file.write("loss_history:" +
                               str(training_log["loss_history"][k])+"\n")
            if best_value > best_metric:
                best_metric = best_value
                best_lr = i
                best_wr = j
    print("best_lr:"+str(best_lr)+" best_wr:"+str(best_wr) +
          " best_metric_value:"+str(best_metric))