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
# from ./pytorch_NEG_loss/NEG_loss/neg import NEG_loss

import time
import os
import copy

import opts

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def main():
    opt = opts.parse_opt()
    data_dir = opt.intput_data_dir

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
    batch_size = opt.batch_size
    image_datasets = {
            x: VisualConceptDataset(
                os.path.join(data_dir, x),
                os.path.join(opt.input_label_dir, x),
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
    log_every = opt.losses_log_every
    print_every = opt.print_every
    checkpoint_path = opt.checkpoint_path
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tb_summary_writer = tb and tb.SummaryWriter(
            op.path.join(checkpoint_path, TIMESTAMP))

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
                    binary_labels = torch.gt(labels, 0).to(torch.int)
                    # print('in:', inputs.size(), labels.size())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = torch.gt(outputs, 0).to(torch.int)
                        # loss only consider label 1 predict and 0 predict ignore
                        # outputs.mul(labels) # label smoothing loss
                        outputs.mul(binary_labels.to(torch.float))
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == binary_labels.data) 

                    '''
                    pred  = [1, 1, 0, 0, 1]
                    label = [1, 0, 0, 1, 1]
                    pred == label = [1, 0, 1, 0, 1]
                    tp = torch.sum([1, 0, 0, 0, 1])
                    fp = torch.sum(pred)
                    prec   = tp/(tp + fp)
                    recall = tp/(tp + fn)
                    '''
                    tmp1 = (preds == binary_labels.data).to(torch.int)
                    tmp2 = binary_labels.mul(tmp1)
                    tp = torch.sum(tmp2).to(torch.float)
                    prec = torch.div(tp, torch.sum(preds).to(torch.float) + 1e-8)
                    recall = torch.div(tp, torch.sum(binary_labels.to(torch.float)) + 1e-8)
                    f1score = torch.div(2 * prec * recall, prec + recall + 1e-8)
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
                        print('TP: {}, Prec: {}, Recall: {} F1_score: {}\n'.format(
                                    tp.data, prec.data, recall.data, f1score.data
                                    ))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_f1score = f1score_sum / dataset_sizes[phase]
                if phase == 'train':
                    add_summary_value(tb_summary_writer, 'train_epoch_loss', epoch_loss, epoch)
                    add_summary_value(tb_summary_writer, 'train_epoch_corrects', epoch_acc, epoch)
                    add_summary_value(tb_summary_writer, 'train_epoch_f1score', epoch_f1score, epoch)
                else:
                    add_summary_value(tb_summary_writer, 'val_epoch_loss', epoch_loss, epoch)
                    add_summary_value(tb_summary_writer, 'val_epoch_corrects', epoch_acc, epoch)
                    add_summary_value(tb_summary_writer, 'val_epoch_f1score', epoch_f1score, epoch)


                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                if phase == 'val' and epoch_f1score > best_acc:
                    print('epoch_f1score:{}, history_best_score:{}', epoch_f1score, best_acc)
                    # best_acc = epoch_acc
                    best_acc = epoch_f1score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                            model.state_dict(), 
                            os.path.join(checkpoint_path, '%s-model-best.pth'%opt.id))
                    torch.save(
                            optimizer.state_dict(), 
                            os.path.join(checkpoint_path, '%s-info-best.path'%opt.id))
                    print("model save to %s/%s-model-best.pth"%(checkpoint_path, opt.id))

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
    num_classes = opt.num_classes
    model_ft.fc = nn.Linear(fc_features, num_classes)

    # model_ft = myResnet(model, num_classes)
    # model_ft = model_ft.to(device)
    model_ft = model_ft.cuda()

    '''
    For example, if a dataset contains 100 positive and 300 negative examples of a single class, 
    then pos_weight for the class should be equal to 300. 
    The loss would act as if the dataset contains 3×100=300 positive examples.
    '''
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.pos_weight).cuda()) # average caption length is 9.5
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # criterion = nn.MultiLabelMarginLoss()
    # criterion = NEG_loss(num_classes, )

    # Observe that all parameters are being optimized
    if opt.optim == 'SGD':
        optimizer_ft = optim.SGD(
                model_ft.parameters(), 
                lr=opt.learning_rate, 
                momentum=opt.momentum)
    elif opt.optim == 'adam':
        optimizer_ft = optim.Adam(
                model_ft.parameters(), 
                lr = 0.001
                )
    else:
        print('%s is not supported yet'%opt.optim)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, 
            step_size=opt.learning_rate_decay_every, 
            gamma=opt.learning_rate_decay_factor)

    model_ft = train_model(
        model_ft, 
        criterion, 
        optimizer_ft, 
        exp_lr_scheduler,
        num_epochs=opt.max_epochs
        )

    torch.save(
            model_ft.state_dict(), 
            os.path.join(checkpoint_path, '%s-model-best.pth'%opt.id))
    print("model save to %s/%s-model-best.pth"%(checkpoint_path, opt.id))

if __name__ == '__main__':
    main()
