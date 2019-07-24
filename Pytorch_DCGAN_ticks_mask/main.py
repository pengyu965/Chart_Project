import os 
import argparse 
import random
import torch

from model import Generator
from model import Discriminator
from unet import UNet

import op

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = "GoogLeNet", 
                        help = "Choose the one model from GoogLeNet and CNN")
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    # parser.add_argument('--load', type=int, default=99,
                        # help='Epoch id of pre-trained model')
    parser.add_argument("--img_path", type=str, help='input_data')
    parser.add_argument("--gt_path", type=str, help='Ground Truth Image')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default = 10, 
                        help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    parser.add_argument('--class_num', type=int, default = 10, 
                        help='class number')
    parser.add_argument("--ngpu", type=int, default = 1,
                        help="The number of avaliable GPU")
    # parser.add_argument('--maxepoch', type=int, default=100,
    #                     help='Max number of epochs for training')

    # parser.add_argument('--im_name', type=str, default='.png',
    #                     help='Part of image name')

    return parser.parse_args()

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        netG = UNet(in_channels = 3, out_channels = 1)
        print(netG)

        operator = op.Operator(netG)
        operator.trainer(FLAGS.img_path, FLAGS.gt_path, FLAGS.bsize, FLAGS.lr, FLAGS.epoch)

    if FLAGS.predict:
        netG = UNet(in_channels = 3, out_channels = 1)
        print(netG)

        if os.path.exists("./weight/9.pt"):
            if torch.cuda.is_available():
                netG.load_state_dict(torch.load("./weight/9.pt"), strict=False)
            else:
                netG.load_state_dict(torch.load("./weight/9.pt", map_location='cpu'), strict=False)
            print("="*6, "\nModel loaded, start prediction", "\n"+"="*6)

        if os.path.exists("./predict_result/") == False:
            os.mkdir("./predict_result/")

        operator = op.Operator(netG)
        operator.predictor(FLAGS.data_path)
