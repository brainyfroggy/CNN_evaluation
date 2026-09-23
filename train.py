from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import cls_eval_model,cls_visualize_model
from dataloader import cls_dataloader
import argparse
# from livelossplot import PlotLosses
from logger import Logger

# plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data/', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='train', type=str, help='the folder of training data')
parser.add_argument('--VAL', default='val', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='test', type=str, help='the folder of test data')


parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')


parser.add_argument('--model_dir', default='./model', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_model_emotion_vgg16_3class_balanced.pth', type=str, help='name of the trained model')

parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion, optimizer, scheduler, num_epochs, device):
    liveloss = PlotLosses()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logger = Logger('./logs')
    for epoch in range(num_epochs):
        logs = {}
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            prefix = ''
            prefix = phase + '_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'accuracy'] = epoch_acc

            # deep copy the model
            if phase == VAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            liveloss.update(logs)
            liveloss.send()

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)

            info = {prefix + 'loss': logs[prefix + 'log loss'], prefix + 'accuracy': logs[prefix + 'accuracy']}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch +1 )

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # 3. Log training images (image summary)
            info = {prefix + 'images': inputs.view(-1, 3, 224, 224)[:10].cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch +1 )

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    dataloaders, dataset_sizes, class_names = cls_dataloader(args.data_dir, args.TRAIN, args.VAL, args.TEST,
                                                         args.batch_size)

    # Load the pretrained model from pytorch
    vgg16 = models.vgg16(pretrained=True)
    print(vgg16.classifier[6].out_features)  # 1000

    # Freeze training for all layers except for the final layer
    for param in vgg16.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    print(vgg16)

    vgg16 = vgg16.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(vgg16.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("Test before training")
    cls_eval_model(dataloaders, dataset_sizes, class_names, args.TEST, vgg16, criterion, device)
    cls_visualize_model(dataloaders, class_names, vgg16, args.TEST, 6, device)  # test before training

    # ---------------- Training-----------------------------#
    vgg16 = train_model(dataloaders, dataset_sizes, args.TRAIN, args.VAL, vgg16, criterion, optimizer_ft, exp_lr_scheduler, args.epoch, device)
    checkpoint = {'model': vgg16,
                  'state_dict': vgg16.state_dict(),
                  'optimizer': optimizer_ft.state_dict()}

    torch.save(checkpoint, os.path.join(args.model_dir, args.model_name))
    # torch.save(vgg16.state_dict(), os.path.join(args.model_dir, args.model_name))

    # ----------------test-----------------------------#
    cls_eval_model(dataloaders, dataset_sizes, class_names, args.TEST, vgg16, criterion, device)
    # ----------------Visualize -----------------------------
    cls_visualize_model(dataloaders, class_names, vgg16, args.TEST, 6, device)


if __name__ == '__main__':

    main()
