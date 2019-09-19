import os 
import argparse 
import random
import torch
import sys
import torch.nn as nn

from model import Generator
from model import Discriminator
from unet import UNet
from torch.utils.tensorboard import SummaryWriter


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
        netG = UNet(in_channels = 3, out_channels = 8)
        # netG = nn.DataParallel(netG).to(device)
        print(netG)

        pretrained_autoencoder = False 

        if pretrained_autoencoder == True:
            print("="*6, "\nTrain the network from loading pretrained model.","\n"+"="*6)
            if os.path.exists("./weight/autoencoder_model.pt") == False:
                print("An error happened, no model file founded here. Stopped!")
                sys.exit()

            if torch.cuda.is_available():
                state_dict = torch.load("./weight/autoencoder_model.pt")
            else:
                state_dict = torch.load("./weight/autoencoder_model.pt", map_location='cpu')

            # netG.load_state_dict(state_dict, strict = False)
            part_state_dict = {}
            for name in state_dict:
                if "inc" in name or "down" in name:
                    part_state_dict[name] = state_dict[name]
            
            netG.load_state_dict(part_state_dict, strict = False)
            for param in netG.inc.parameters():
                param.requires_grad = False
            for param in netG.down1.parameters():
                param.requires_grad = False
            for param in netG.down2.parameters():
                param.requires_grad = False
            for param in netG.down3.parameters():
                param.requires_grad = False
            for param in netG.down4.parameters():
                param.requires_grad = False

        else:

            if os.path.exists("./weight/model.pt"):
                if torch.cuda.is_available():
                    netG.load_state_dict(torch.load("./weight/model.pt"), strict=True)
                else:
                    netG.load_state_dict(torch.load("./weight/model.pt", map_location='cpu'), strict=True)
                print("="*6, "\nModel loaded, start retraining", "\n"+"="*6)
            else:
                print("="*6, "\nModel isn't found, train the network from begining.","\n"+"="*6)
        
        writer = SummaryWriter("./logs/")            

        operator = op.Operator(netG)
        operator.trainer(FLAGS.img_path, FLAGS.gt_path, FLAGS.bsize, FLAGS.lr, FLAGS.epoch, writer = writer)
        writer.close()


    if FLAGS.eval:
        netG = UNet(in_channels = 3, out_channels = 8)
        # netG = nn.DataParallel(netG).to(device)
        print(netG)

        if os.path.exists("./weight/model.pt"):
            if torch.cuda.is_available():
                netG.load_state_dict(torch.load("./weight/model.pt"), strict=True)
            else:
                netG.load_state_dict(torch.load("./weight/model.pt", map_location='cpu'), strict=True)
            print("="*6, "\nModel loaded, start evaluation", "\n"+"="*6)
        else:
            print("="*6, "\nModel isn't found, train the network first.","\n"+"="*6)
            sys.exit()

        operator = op.Operator(netG)
        with torch.no_grad():
            operator.validator(FLAGS.img_path, FLAGS.gt_path, global_step = "validation", train_regression = True)

    if FLAGS.predict:
        netG = UNet(in_channels = 3, out_channels = 8)
        # netG = nn.DataParallel(netG).to(device)
        print(netG)

        if os.path.exists("./weight/model.pt"):
            if torch.cuda.is_available():
                netG.load_state_dict(torch.load("./weight/model.pt"), strict=True)
            else:
                netG.load_state_dict(torch.load("./weight/model.pt", map_location='cpu'), strict=True)
            print("="*6, "\nModel loaded, start prediction", "\n"+"="*6)
        else:
            print("="*6, "\nModel isn't found, train the network first.","\n"+"="*6)
            sys.exit()

        if os.path.exists("./predict_result/") == False:
            os.mkdir("./predict_result/")

        operator = op.Operator(netG)
        operator.predictor(FLAGS.img_path, visualize = False)
