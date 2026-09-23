from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torchvision import models
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import time
import os
import copy
from utils import reg_eval_model,reg_visualize_model, load_checkpoint
from dataloader import reg_dataloader
import argparse
# from livelossplot import PlotLosses
from logger import Logger
from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from torchsummary import summary
# NOTE: `Amygdala2` does not exist in network.py; the "LA+CE" (lateral + central
# nucleus) architecture it names lives there as `Amygdala_lowroad`, with the same
# lowfea_VGGlayer/highfea_VGGlayer arguments used below. Aliased here so the rest
# of this file (option 2 of --model_to_run) needs no further changes.
from network import VGGReg, Amygdala, Amygdala_lowroad as Amygdala2
# plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='/data/DATA/VAM/data/', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='train', type=str, help='the folder of training data')
parser.add_argument('--VAL', default='val', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='test', type=str, help='the folder of test data')

parser.add_argument('--csv_train', default='/data/DATA/VAM/data/Kim_train.csv', type=str, help='the path of training data csv file')
parser.add_argument('--csv_val', default='/data/DATA/VAM/data/Kim_test.csv', type=str, help='the path of training data csv file')
parser.add_argument('--csv_test', default='/data/DATA/VAM/data/Kim_test.csv', type=str, help='the path of training data csv file')


parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
#smaller lr is important or the output will be nan
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for SGD')


parser.add_argument('--model_dir', default='./model', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_model_emotion_vgg16_regression_Amygdala2_0508_50epoch_lr3_32bs_ckv_alex_20PM.pth', type=str, help='name of the trained model')
parser.add_argument('--n_layers', default=2, type=int, help='the number of layers in Amygdala')
parser.add_argument('--resume', default=None, type=str, help='the path to checkpoint')
parser.add_argument('--start_epoch', default=29, type=int, metavar='N', help='manual epoch number (useful on restarts)')


parser.add_argument('--model_to_run', default=2, type=int, help='0: VGG (replace the last layer with one unit); 1: Amygdala(VGG + n fc layers);  2: Amygdala2 (LA+CE)' )

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion, optimizer, scheduler, num_epochs, device, start_epoch=0):
    args = parser.parse_args()
    # liveloss = PlotLosses()
    since = time.time()

    labels_list = []
    preds_list = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_R = 0.0
    best_epoch =0
    best_loss = 10
    logger = Logger('./logs/Amygdala2_0508_50epoch_lr3_32bs_ckv_alex_20PM/')
    for epoch in range(start_epoch, num_epochs):
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
                    # learning rate has to be smaller than 1e-3 otherwise the output will be nan
                    loss = criterion(outputs.view(labels.size()), labels.float())



                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        # summary(model, (3, 224, 224))
                        #
                        # for name, param in model.named_parameters():
                        #     if param.requires_grad:
                        #         print(name)
                        optimizer.step()

                    # statistics
                    running_loss += loss.data
                    labels_list.extend(labels.cpu().numpy())
                    preds = outputs.reshape(labels.shape).cpu().detach().numpy()
                    preds_list.extend(preds)

                    # clean the cache
                    # del inputs, labels, outputs, loss, preds
                    # torch.cuda.empty_cache()

            if phase == TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_R, _ = pearsonr(labels_list, preds_list)
            epoch_R2 = np.around(epoch_R ** 2,2)


            print('{} Loss: {:.4f} R: {:.4f} R2: {:.4f}'.format(
                phase, epoch_loss, epoch_R, epoch_R2))


            prefix = phase + '_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'correlation'] = epoch_R

            # deep copy the model
            if phase == VAL and epoch_R > best_R:
                best_R = epoch_R
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {'model': model,
                              'epoch': best_epoch,
                              'best_R': best_R,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}

                torch.save(checkpoint, os.path.join(args.model_dir, args.model_name))

                print('Best epoch: {:} Best R: {:.4f} '.format(
                    best_epoch, best_R))
            # liveloss.update(logs)
            # liveloss.send()

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)

            info = {prefix + 'loss': logs[prefix + 'log loss'], prefix + 'correlation': logs[prefix + 'correlation']}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch +1 )

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                if value.requires_grad:
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # 3. Log training images (image summary)
            info = {prefix + 'images': inputs.view(-1, 3, 224, 224)[:10].cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, (images * 255).astype(np.uint8), epoch +1 )

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val R: {:4f}'.format(best_R))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    dataloaders, dataset_sizes = reg_dataloader(args.data_dir, args.TRAIN, args.VAL, args.TEST,
                                                         args.csv_train, args.csv_val, args.csv_test,
                                                         args.batch_size)

    if args.model_to_run == 0:
        model = VGGReg()
    elif args.model_to_run == 1:
        model = Amygdala(args.n_layers)
    else:
        # `input_size_CE` isn't an Amygdala_lowroad constructor argument (it's fixed
        # internally to 4096 + 512), so it's dropped here rather than passed through.
        model = Amygdala2(lowfea_VGGlayer=0, highfea_VGGlayer=36)


    criterion = nn.MSELoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # # Map model to be loaded to specified single gpu.
            # loc = 'cuda:{}'.format(args.gpu_ids)
            checkpoint = torch.load(args.resume)
            if "epoch" in checkpoint:
                start_epoch = checkpoint['epoch']
            else:
                start_epoch = args.start_epoch
            model = checkpoint['model']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_ft.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print(model)

    model = model.to(device)

    summary(model, (3, 224, 224))


    print("Test before training")
    reg_eval_model(dataloaders, dataset_sizes, args.TEST, model, criterion, device)
    reg_visualize_model(dataloaders, model, args.TEST, 6, device)  # test before training

    # ---------------- Training-----------------------------#
    model = train_model(dataloaders, dataset_sizes, args.TRAIN, args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler, args.epoch, device, start_epoch)
    # checkpoint = {'model': model,
    #               'state_dict': model.state_dict(),
    #               'optimizer': optimizer_ft.state_dict()}

    # torch.save(checkpoint, os.path.join(args.model_dir, args.model_name))
    # torch.save(vgg16.state_dict(), os.path.join(args.model_dir, args.model_name))

    # ----------------test-----------------------------#
    reg_eval_model(dataloaders, dataset_sizes,  args.TEST, model, criterion, device)
    # ----------------Visualize -----------------------------
    reg_visualize_model(dataloaders, model, args.TEST, 6, device)


if __name__ == '__main__':

    main()
