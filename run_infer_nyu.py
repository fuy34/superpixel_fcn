import argparse
import os

import torch.backends.cudnn as cudnn

import models
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave

from loss import *
import time
import random


'''
Infer from nyu dataset:
author:Fengting Yang 
last modification: Mar.14th 2019

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='', help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model', default= './pretrain_ckpt/SpixelNet_bsd_ckpt.tar')
parser.add_argument('--output', metavar='DIR', default= '' , help='path to output folder')

parser.add_argument('--downsize', default=16, type=float,help='superpixel grid cell, must be same as training setting')

parser.add_argument('-nw', '--num_threads', default=1, type=int,  help='num_threads')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

# nyu only has one type
parser.add_argument('--input_img_height', '-v_imgH', default=480,  type=int, help='img height_must be 16*n')
parser.add_argument('--input_img_width', '-v_imgW', default=640,   type=int, help='img width must be 16*n')

args = parser.parse_args()
args.test_list = args.data_dir + '/nyuv2_test_subset.txt'

random.seed(100)
@torch.no_grad()
def test(args, model, img_paths, save_path, spixeIds, idx,scale):
      # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    img_ = imread(load_path)
    H_, W_, _ = img_.shape
    img = cv2.resize(img_, (int( args.input_img_width), int( args.input_img_height)), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)

    # The orignal sz of nyu test set 448*608
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( int(448), int(608)), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= 1200*scale*scale,  b_enforce_connect=True)

    # ************************ Save all result********************************************
    #save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))


    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
      # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i',delimiter=",")


    if idx % 10 == 0:
        print("processing %d"%idx)

    return toc

def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    input_img_height = args.input_img_height
    input_img_width = args.input_img_width

    for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        assert (input_img_height * scale % 16 == 0 and input_img_width * scale % 16 == 0)
        save_path = args.output + '/SPixelNet_nSpixel_{0}'.format(int(input_img_height/16 * scale * input_img_width /16 * scale))

        args.input_img_height, args.input_img_width = input_img_height * scale, input_img_width * scale

        print('=> will save everything to {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        tst_lst = []
        with open(args.test_list, 'r') as tf:
            img_path = tf.readlines()
            for id in img_path:
                img_path = os.path.join(data_dir,'img/%.5d.jpg'%int(id[:-1]))
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the NYUv2 dataset as README states and provide the correct dataset path.')
                    exit(1)
                tst_lst.append(img_path)

        print('{} samples found'.format(len(tst_lst)))

        # create model
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained model '{}'".format(network_data['arch']))
        model = models.__dict__[network_data['arch']]( data = network_data).cuda()
        model.eval()
        args.arch = network_data['arch']
        cudnn.benchmark = True

        spixlId, XY_feat_stack = init_spixel_grid(args, b_train=False)

        mean_time = 0
        for n in range(len(tst_lst)):
          time = test(args, model, tst_lst, save_path, spixlId, n,scale)
          mean_time += time
        print("avg_time per img: %.3f"%(mean_time/len(tst_lst)))

if __name__ == '__main__':
    main()
