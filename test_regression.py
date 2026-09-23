from __future__ import print_function, division
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import reg_eval_model, reg_visualize_model,load_checkpoint
from _dataloader import reg_dataloader_test
from torchsummary import summary
import argparse
import time
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--mode', default='server', type=str, help='mode') #MODIFIED BY YUJUN

parser.add_argument('--data_dir', default='F:/yujun/projects/data/IAPS_AI', type=str, help='the data root folder')                         # Variable to change
# parser.add_argument('--data_dir', default='F:/yujun/projects/data/', type=str, help='the data root folder')    
# parser.add_argument('--TRAIN', default='train', type=str, help='the folder of training data')
# parser.add_argument('--VAL', default='val', type=str, help='the folder of validation data')
# parser.add_argument('--TEST', default='IAPS1182/IAPS1182', type=str, help='the folder of test data')
parser.add_argument('--TEST', default='IAPS_60', type=str, help='the folder of test data')

# parser.add_argument('--csv_train', default='./data/Kim/Kim_train.csv', type=str, help='the path of training data csv file')
# parser.add_argument('--csv_val', default='./data/Kim/Kim_test.csv', type=str, help='the path of training data csv file')
# parser.add_argument('--csv_test', default='F:/yujun/projects/data/IAPS1182/IAPS1182.csv', type=str, help='the path of training data csv file')
parser.add_argument('--csv_test', default='F:/yujun/projects/data/IAPS_AI/IAPS_60_valence.csv', type=str, help='the path of training data csv file')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')

parser.add_argument('--model_dir', default='./model', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_model_emotion_regression_Amygdala_0521_50epoch_lr4_128bs_7PM_ckvideo_middleframe_train_epoch3_best_R0.3522.pth', type=str, help='name of the trained model')
# parser.add_argument('--model_name', default='best_model_emotion_regression_VGG_0522_50epoch_lr4_128bs_23M_ckvideo_middleframe_train_normalization_epoch2_best_R0.5468.pth', type=str, help='name of the trained model')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, dataset_sizes = reg_dataloader_test(args.data_dir, args.TEST, args.csv_test, args.batch_size)

    # Load the  model
    model = load_checkpoint(os.path.join(args.model_dir, args.model_name))
    print("model = ", model)

    model = model.to(device)
    summary(model, (3, 224, 224))

    criterion = nn.MSELoss()


    for name, module in model.named_modules():
        print(name)

    # ----------------test-----------------------------#
    reg_eval_model(dataloaders, dataset_sizes, args.TEST, model, criterion, device)
    # ----------------Visualize -----------------------------
    # reg_visualize_model(dataloaders, model, args.TEST, 6, device)

