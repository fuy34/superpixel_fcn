import os
import numpy as np
import cv2
from scipy.io import loadmat
import argparse
from glob import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="", help="where the filtered dataset is stored")
parser.add_argument("--dump_root", type=str, default="", help="Where to dump the data")
parser.add_argument("--b_filter", type=bool, default=False, help="we do not use this in our paper")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

'''
Extract each pair of image and label from .mat to generagte the data for TRAINING and VALIDATION
Please generate TEST data with pre_process_bsd500_ori_sz.py in the same folder 

We follow the SSN configuration to discard all samples that have more than 50 classes in their segments, and 
we use the exactaly same train, val, and test list as SSN, see the train/val/test.txt in the data_preprocessing folder for details

  
author: Fengting Yang 
March. 1st 2019 
'''

def make_dataset(dir):
    cwd = os.getcwd()
    train_list_path = cwd + '/train.txt'
    val_list_path =  cwd + '/val.txt'
    train_list = []
    val_list = []

    try:
        with open(train_list_path, 'r') as tf:
            train_list_0 = tf.readlines()
            for path in train_list_0:
                img_path = os.path.join(dir, 'BSR/BSDS500/data/images/train', path[:-1]+ '.jpg' )
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                train_list.append(img_path)

        with open (val_list_path, 'r') as vf:
            val_list_0 = vf.readlines()
            for path in val_list_0:
                img_path = os.path.join(dir, 'BSR/BSDS500/data/images/val', path[:-1]+ '.jpg')
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                val_list.append(img_path)


    except IOError:
        print ('Error No avaliable list ')
        return

    return train_list, val_list

def convert_label(label):

    problabel = np.zeros(( label.shape[0], label.shape[1], 50)).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            print('give up sample because label shape is larger than 50: {0}'.format(np.unique(label).shape))
            break
        else:
            problabel[ :, :, ct] = (label == t) #one hot
        ct = ct + 1

    label2 = np.squeeze(np.argmax(problabel, axis = -1)) #squashed label e.g. [1. 3. 9, 10] --> [0,1,2,3], (h*w)

    return label2, problabel

def BSD_loader(path_imgs, path_label, b_filter=False):

    img_ = cv2.imread(path_imgs)

    # origin size 481*321 or 321*481
    H_, W_, _ = img_.shape

    # crop to 16*n size
    if H_ == 321 and W_ == 481:
        img = img_[:320, :480, :]
    elif H_ == 481 and W_ == 321:
        img = img_[:480, :320, :]
    else:
        print('It is not BSDS500 images')
        exit(1)

    if b_filter:
        img = cv2.bilateralFilter(img, 5, 75, 75)

    gtseg_lst = []

    gtseg_all = loadmat(path_label)
    for t in range(len(gtseg_all['groundTruth'][0])):
        gtseg = gtseg_all['groundTruth'][0][t][0][0][0]

        label_, _ = convert_label(gtseg)
        if H_ == 321 and W_ == 481:
            label = label_[:320, :480]
        elif H_ == 481 and W_ == 321:
            label = label_[:480, :320]

        gtseg_lst.append(label)

    return img, gtseg_lst

def dump_example(n, n_total, dataType, img_path):
    global args
    if n % 100 == 0:
        print('Progress {0} {1}/{2}....' .format (dataType,n, n_total))

    img, label_lst = BSD_loader(img_path, img_path.replace('images', 'groundTruth')[:-4]+'.mat', b_filter=args.b_filter)

    if args.b_filter:
        dump_dir = os.path.join(args.dump_root, dataType + '_b_filter_' + str(args.b_filter))
    else:
        dump_dir = os.path.join(args.dump_root, dataType)

    if not os.path.isdir(dump_dir):
        try:
            os.makedirs(dump_dir)
        except OSError:
            if not os.path.isdir(dump_dir):
                raise

    img_name = os.path.basename(img_path)[:-4]
    for k, label in enumerate(label_lst):
        # save images
        dump_img_file = os.path.join(dump_dir,  '{0}_{1}_img.jpg' .format(img_name, k))
        cv2.imwrite(dump_img_file, img.astype(np.uint8))

        # save label
        dump_label_file = os.path.join(dump_dir, '{0}_{1}_label.png' .format(img_name, k))
        cv2.imwrite(dump_label_file, label.astype(np.uint8))

        # save label viz, uncomment if needed 
        # if not os.path.isdir(os.path.join(dump_dir,'label_viz')):
        #     os.makedirs(os.path.join(dump_dir,'label_viz'))
        # dump_label_viz = os.path.join(dump_dir, 'label_viz',  '{0}_{1}_label_viz.png'.format(img_name, k))
        # plt.imshow(label)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig(dump_label_viz,bbox_inches='tight',pad_inches=0)
        # plt.close()


def main():
    datadir = args.dataset
    train_list, val_list = make_dataset(datadir)

    dump_pth = os.path.abspath(args.dump_root)
    print("data will be saved to {}".format(dump_pth))
    # for debug only
    # for n, train_samp in enumerate(train_list):
    #     dump_example(n, len(train_list),'train', train_samp)

    # mutil-thread running for speed
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, len(train_list),'train', train_samp) for n, train_samp in enumerate(train_list))
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, len(train_list),'val', val_samp) for n, val_samp in enumerate(val_list))

    with open(dump_pth + '/train.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_pth, 'train', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')

    with open(dump_pth+ '/val.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_pth, 'val', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')

if __name__ == '__main__':
    main()