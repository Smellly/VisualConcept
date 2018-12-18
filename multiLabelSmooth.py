# -*- encoding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from visualConceptDataset import VisualConceptDataset
from myNets import myResnet

import time
import os
import copy

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def main():
    data_dir = '/media/disk0/jay/workspace/visual-concept/imgs'

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    batch_size = 64
    image_datasets = {
            x: VisualConceptDataset(
                os.path.join(data_dir, x),
                data_transforms[x]) 
            for x in ['train', 'val']
            }
    dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=4)
            for x in ['train', 'val']
            }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    log_every = 10
    print_every = 10
    checkpoint_path = 'slogs'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    tb_summary_writer = tb and tb.SummaryWriter(checkpoint_path)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        iteration = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 100)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                f1score_sum = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # inputs = inputs.to(device)
                    inputs = inputs.cuda()
                    # labels = labels.float().to(device)
                    labels = labels.to(torch.float).cuda()
                    # print('in:', inputs.size(), labels.size())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = torch.gt(outputs, 0).to(torch.int8)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.to(torch.int8).data) 

                    '''
                    pred  = [1, 1, 0, 0, 1]
                    label = [1, 0, 0, 1, 1]
                    pred == label = [1, 0, 1, 0, 1]
                    tp = torch.sum([1, 0, 0, 0, 1])
                    fp = torch.sum(pred)
                    prec   = tp/(tp + fp)
                    recall = tp/(tp + fn)
                    '''
                    tmp1 = (preds == labels.to(torch.int8).data).to(torch.int8)
                    tmp2 = torch.gt(labels,0).to(torch.int8).mul(tmp1)
                    tp = torch.sum(tmp2).to(torch.float)
                    prec = torch.div(tp, torch.sum(preds).to(torch.float) + 1e-8)
                    recall = torch.div(tp, torch.sum(labels) + 1e-8))
                    f1score = torch.div(2 * prec * recall, prec + recall)
                    f1score_sum += f1score

                    # print('pred:', preds.size(), labels.size())

                    iteration += 1
                    add_summary_value(tb_summary_writer, 'train_loss', loss, iteration)
                    add_summary_value(tb_summary_writer, 'running_loss', running_loss/iteration, iteration)
                    add_summary_value(tb_summary_writer, 'running_corrects', running_corrects/iteration, iteration)
                    add_summary_value(tb_summary_writer, 'running_tp', tp, iteration)
                    add_summary_value(tb_summary_writer, 'running_prec', prec, iteration)
                    add_summary_value(tb_summary_writer, 'running_recall', recall, iteration)
                    add_summary_value(tb_summary_writer, 'running_f1score', f1score, iteration)

                    if (iteration % print_every == 0):
                        print('{} : Epoch {} Iteration {} Loss: {:.4f}/10000 running_loss: {:.4f}, Acc: {:.4f}'.format(
                                    phase, epoch, iteration, loss*10000, running_loss/iteration, 
                                    running_corrects/batch_size))
                        print('TP: {}, Prec: {}, Recall: {} F1_score: {}'.format(
                                    tp.data, prec.data, recall.data, f1score.data
                                    ))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_f1score = f1score_sum / dataset_sizes[phase]
                if phase == 'train':
                    add_summary_value(tb_summary_writer, 'train_epoch_loss', epoch_loss, iteration)
                    add_summary_value(tb_summary_writer, 'train_epoch_corrects', epoch_acc, iteration)
                else:
                    add_summary_value(tb_summary_writer, 'val_epoch_loss', epoch_loss, iteration)
                    add_summary_value(tb_summary_writer, 'val_epoch_corrects', epoch_acc, iteration)


                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                if phase == 'val' and epoch_f1score > best_acc:
                    # best_acc = epoch_acc
                    best_acc = epoch_f1score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                            model.state_dict(), 
                            os.path.join(checkpoint_path, 'smodel-best.pth'))
                    torch.save(
                            optimizer.state_dict(), 
                            os.path.join(checkpoint_path, 'sinfo-best.path'))
                    print("model save to %s/model-best.pth"%checkpoint_path)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # 调用模型
    model_ft = models.resnet101(pretrained=True)
    # 提取fc层中固定的参数
    fc_features = model_ft.fc.in_features
    # 修改类别为 vocab_size
    model_ft.fc = nn.Linear(fc_features, 9360)

    # model_ft = myResnet(model, 9360)
    # model_ft = model_ft.to(device)
    model_ft = model_ft.cuda()

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # criterion = nn.MultiLabelMarginLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(
            model_ft.parameters(), 
            lr=0.1, 
            momentum=0.9)
    '''
    optimizer_ft = optim.Adam(
            model_ft.parameters(), 
            lr = 0.001
            )
    '''
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft, 
        criterion, 
        optimizer_ft, 
        exp_lr_scheduler,
        num_epochs=20
        )

    torch.save(
            model_ft.state_dict(), 
            os.path.join(checkpoint_path, 'smodel-best.pth'))
    print("model save to %s/model-best.pth"%checkpoint_path)

if __name__ == '__main__':
    main()
