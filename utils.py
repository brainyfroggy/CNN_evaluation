import matplotlib
import glob
matplotlib.use('Agg')

import torchvision
import torch
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr
import os
import seaborn as sns
import warnings
from pathlib import Path
import shutil
import pandas as pd


warnings.filterwarnings("ignore")

def get_filenames(dir):
    names = []
    imgs = os.path.join(dir,'*.jpg')
    imgs = sorted(glob.glob(imgs)) 
    for img in imgs:
        name = Path(img).stem
        names.append(name)
    return names

def get_dirnames(dir):
    names = []
    for name in os.listdir(dir):
        names.append(name)
    names.sort()
    return names

def gen_nsubset(input_dir_fullname, input_dir, n_subset, output_dir):
    fnames = get_filenames(input_dir_fullname)
    fname_select = random.choice(fnames)
    shutil.copy(os.path.join(input_dir_fullname,'{}.jpg'.format(fname_select)),os.path.join(output_dir,'{}.jpg'.format(fname_select)))
    # print("moved n_subset image(s) from {} to {}".format(input_dir_fullname, output_dir))

def gen_select_csv(image_folder, select_names, input_csv, output_csv):
    # find common image names
    df_select = pd.DataFrame(select_names, columns=['id'], dtype = 'string')
    df_IAPS = pd.read_csv(input_csv, names=['id','file_type','valence'],dtype = 'string')
    df_IAPS_select = pd.merge(df_select, df_IAPS, on=['id'], how='inner')
    
    df_input = pd.DataFrame(columns = ['id','file_type','valence'])

    total_number_files = 0

    for i in range(0, len(df_IAPS_select)): #928
        img_dir_name = df_IAPS_select.loc[i]['id']
        # n = len() # number of rand images in one image dir
        randImagelist = os.listdir(os.path.join(image_folder,img_dir_name)) # dir is your directory path
        number_files = len(randImagelist)
        for j in range(0,number_files):
            id = df_IAPS_select.iloc[i,0] + '/'+ Path(randImagelist[j]).stem
            valence = df_IAPS_select.iloc[i,2]
            index = total_number_files+j
            df_input.loc[index] = [id,'jpg',valence]
        total_number_files = total_number_files + number_files

    df_input.to_csv(output_csv, header=None, index = False)

def gen_select_csv_simple(select_names, input_csv, output_csv):
    # find common image names
    df_select = pd.DataFrame(select_names, columns=['id'], dtype = 'string')
    df_IAPS = pd.read_csv(input_csv, names=['id_join','file_type','valence'],dtype = 'string')
    df_select['id_join']=df_select.apply(lambda x: x['id'][:len(x['id'])-2],axis = 1)
    df_IAPS_select = pd.merge(df_select, df_IAPS, on=['id_join'], how='inner')
    
    df_IAPS_select.to_csv(output_csv, header=None, index = False)

def imshow(inp, marker, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    fig.savefig('result/visual_result_' + '_' + now + '_' + marker + '.pdf')
    plt.pause(0.001)


def show_databatch(class_names, inputs, classes, marker):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, marker, title=[class_names[x] for x in classes])


def show_databatch_regression(inputs, classes, marker):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, marker, title=[round(x, 2) for x in classes.numpy()])


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # SMALL_SIZE = 8
    # matplotlib.rc('font', size=SMALL_SIZE)
    # matplotlib.rc('axes', titlesize=SMALL_SIZE)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=3)
    plt.ylabel('True label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label \nAccuracy={:0.4f}'.format(accuracy))

    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    fig.savefig('result/conf_matrix_result_' + '_' + now + '.pdf')

    plt.show()


def plot_precision_recall(n_classes, Y_test, y_score):
    ###############################################################################
    # The average precision score in multi-label settings

    # np_labels_bn= label_binarize(np_labels, classes=[0, 1, 2])
    # np_preds_bn = label_binarize(filter_preds, classes=[0, 1, 2])

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[i], y_score[i])
        average_precision[i] = average_precision_score(Y_test[i], y_score[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    ###############################################################################
    # Plot the micro-averaged Precision-Recall curve
    # ...............................................
    #

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    ###############################################################################
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # .............................................................
    #
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()


def filter_array(np_labels, np_preds, n_classes):
    results_labels = {}
    results_preds = {}

    for i in range(n_classes):
        condition = (np_labels == i)
        results_labels[i] = label_binarize(np.extract(condition, np_labels), classes=[0, 1, 2])
        results_preds[i] = label_binarize(np.extract(condition, np_preds), classes=[0, 1, 2])

    return results_labels, results_preds


def plot_correaltion(y, y_cv, score_cv_r, score_cv_r2, mse_cv):
    # Fit a line to the C vs response

    z = np.polyfit(y, y_cv, 1)
    sns.set()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_cv, c='red', edgecolors='k')
    # Plot the best fit line
    ax.plot(y, np.polyval(z, y), c='blue', linewidth=1)

    plt.title('$R^{2}$: %5.4f' % score_cv_r2 + ' \n  R: %5.4f' % score_cv_r + '; MSE: %5.4f' % mse_cv)

    plt.ylabel('Predicted valence rating')
    plt.xlabel('Measured valence rating')

    # set axes range
    plt.xlim(1, 9)
    plt.ylim(1, 9)

    plt.show()
    now = datetime.now()
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    # fig.savefig('result/regression_plot' + '_' + str(now) + '_' + 'R-%5.4f' % score_cv_r + '.pdf')
    fig.savefig('result/regression_plot' +  '.pdf')


# for classification
def cls_visualize_model(dataloaders, class_names, vgg, test_folder, num_images, device):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[test_folder]):
        inputs, labels = data
        size = inputs.size()[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]

        print("Ground truth:")
        show_databatch(class_names, inputs.data.cpu(), labels.data.cpu(), 'Ground truth')
        print("Prediction:")
        show_databatch(class_names, inputs.data.cpu(), predicted_labels, 'Prediction')

        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


# for classification
def cls_eval_model(dataloaders, dataset_sizes, class_names, test_folder, vgg, criterion, device):
    since = time.time()
    loss_test = 0
    acc_test = 0
    labels_list = []
    preds_list = []

    test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)
        print(acc_test)

        labels_list.extend(labels.cpu().numpy())
        preds_list.extend(preds.cpu().numpy())

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    # calculate accuracy
    avg_loss = loss_test / dataset_sizes[test_folder]
    avg_acc = acc_test.double() / dataset_sizes[test_folder]
    # plot confusion matrix
    conf_matr = confusion_matrix(labels_list, preds_list)
    plot_confusion_matrix(conf_matr, normalize=False, target_names=class_names, title="Confusion Matrix")

    # plot precision and recall
    # np_labels = np.asarray(labels_list)
    # np_preds = np.asarray(preds_list)
    # n_classes = len(class_names)
    # filter_labels, filter_preds = filter_array(np_labels, np_preds, n_classes)

    # plot_precision_recall(n_classes, filter_labels, filter_preds)

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('Confussion matrix:')
    print(conf_matr)
    print('-' * 10)
    print(classification_report(labels_list, preds_list, digits=4))
    print('-' * 10)


# for regression
def reg_visualize_model(dataloaders, vgg, test_folder, num_images, device):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[test_folder]):
        _, inputs, labels = data
        size = inputs.size()[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        # _, preds = torch.max(outputs.data, 1)
        predicted_labels = outputs.view(labels.size())  # [outputs[j] for j in range(inputs.size()[0])]

        print("Ground truth:")
        if num_images <= inputs.shape[0]:
            show_databatch_regression(inputs.data[:num_images].cpu(), labels.data[:num_images].cpu(), 'Ground truth')
        else:
            show_databatch_regression(inputs.data.cpu(), labels.data.cpu(), 'Ground truth')
        print("Prediction:")
        if num_images <= inputs.shape[0]:
            show_databatch_regression(inputs.data[:num_images].cpu(), predicted_labels[:num_images].detach().cpu(),
                                      'Prediction')
        else:
            show_databatch_regression(inputs.data.cpu(), predicted_labels, 'Prediction')

        del inputs, labels, outputs, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


# for regression
def reg_eval_model(dataloaders, dataset_sizes, test_folder, vgg, criterion, device):
    since = time.time()
    loss_test = 0
    labels_list = []
    preds_list = []

    test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        if i % 100 == 0:
            print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)
            # print(i, data['image'].size(), data['lable'].size())

        _, inputs, labels = data

        labels = labels.to(device)
        inputs = inputs.to(device)

        vgg.train(False)
        vgg.eval()
        outputs = vgg(inputs)
        #outputs = 1 + outputs * (9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
        loss = criterion(outputs.view(labels.size()), labels.float())

        loss_test += loss.data
 
        print(i + 1, '{:.4f}'.format(loss.cpu().item()))

        # here taking lots of time to fix the GPU out of Memory issue: need to  be numpy()
        labels_list.extend(labels.cpu().numpy())
        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
        preds_list.extend(preds)

        # clean the cache
        del inputs, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    # calculate correalation R
    avg_loss = loss_test / dataset_sizes[test_folder]
    score_c_r, _ = pearsonr(labels_list, preds_list)
    score_cv_r2 = np.around(score_c_r ** 2, 2)  # r2_score(labels_list, preds_list)

    # mse_loss = mean_squared_error(labels_list, preds_list)

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg MSE loss (test): {:.4f}".format(avg_loss))
    print("R (test): {:.4f}".format(score_c_r))
    print('R2 : %5.4f' % score_cv_r2)
    # print('MSE CV: %5.4f' % mse_loss)

    # plot correatlion
    plot_correaltion(labels_list, preds_list, score_c_r, score_cv_r2, avg_loss)

    print('-' * 10)

    pd.DataFrame(labels_list).to_csv("./result/labels_list.csv")
    pd.DataFrame(preds_list).to_csv("./result/preds_list.csv")



# def extract_features(images, result_folder = 'result/'):


def save_checkpoint(best_R, best_loss, best_epoch, model, optimizer, model_name):
    checkpoint = {'model': model,
                  'epoch': best_epoch,
                  'best_R': best_R,
                  'best_loss': best_loss,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, model_name)


# for test
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=torch.device('cpu') )

    if 'best_R' in checkpoint:
        print("Best R: {:.4f}".format(checkpoint['best_R']))
    if 'epoch' in checkpoint:
        print("Best epoch: {:}".format(checkpoint['epoch']))

    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def load_checkpoint2(model,filepath, tuning_layer=None, istuning=False):

    checkpoint = torch.load(filepath, map_location='cuda:0')
    state_dict = checkpoint['state_dict']

    if istuning and tuning_layer is not None:

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            layer_idx = k.split('.')[-2]
            if int(layer_idx) > tuning_layer:
                k = k.replace('features', 'rest_conv_part_net')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
    else:
        # model = checkpoint['model']
        model.load_state_dict(state_dict)

    for parameter in model.parameters():
        parameter.requires_grad = False

    # if 'best_R' in checkpoint:
    #     print("Best R: {:.4f}".format(checkpoint['best_R']))
    # if 'best_per' in checkpoint:
    #     print("Best peformance: {:.4f}".format(checkpoint['best_per']))
    # if 'best_loss' in checkpoint:
    #     print("Best loss: {:.4f}".format(checkpoint['best_loss']))
    # if 'epoch' in checkpoint:
    #     print("Best epoch: {:}".format(checkpoint['epoch']))

    model.eval()
    return model


from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


