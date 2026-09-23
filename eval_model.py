import os
import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import reg_eval_model, load_checkpoint
from _dataloader import reg_dataloader_test
from torchsummary import summary

plt.ion()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--mode', default='server', type=str, help='mode')      #MODIFIED BY YUJUN

parser.add_argument('--data_dir', default='./data/', type=str, help='the data root folder')                         # Variable to change
parser.add_argument('--test_subset', default='40_rand_min_occ/images', type=str, help='the folder of subset test data')
parser.add_argument('--csv_select_subset', default='./data/IAPS_select_all.csv', type=str, help='the path of testing data csv file')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--model_dir', default='./model_peng', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_model_emotion_regression_Amygdala_0521_50epoch_lr4_128bs_7PM_ckvideo_middleframe_train_epoch3_best_R0.3522.pth', type=str, help='name of the trained model')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


def evaluate_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load all parameters
    args = parser.parse_args()
    
    # generate dataloader
    dataloaders, dataset_sizes = reg_dataloader_test(args.data_dir, args.test_subset, args.csv_select_subset, args.batch_size)

    # Load the model
    model_path = os.path.join(args.model_dir, args.model_name)
    model = load_checkpoint(model_path)
    # print("model = ", model)

    # send model to the device, cpu or gpu based on device value
    model = model.to(device)
    
    # print model summary
    summary(model, (3, 224, 224))

    # define model measurement criterion
    criterion = nn.MSELoss()

    # print model structure
    # for name, module in model.named_modules():
    #     print(name)

    # -------------------- test on model ------------------------#
    reg_eval_model(dataloaders, dataset_sizes, args.test_subset, model, criterion, device)
    
    # ---------------- visualize result -----------------------------
    # reg_visualize_model(dataloaders, model, args.TEST, 6, device)


    