import os 
import argparse 
import random
import torch
import sys
import torch.nn as nn

from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision


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

    # parser.add_argument('--maxepoch', type=int, default=100,
    #                     help='Max number of epochs for training')

    # parser.add_argument('--im_name', type=str, default='.png',
    #                     help='Part of image name')

    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        
        num_classes = 6

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 512
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        print(model)

        # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # backbone.out_channels = 1280
        # anchor_generator = AnchorGenerator(sizes=((2,3,5,10,15,20),),
        #     aspect_ratios=((0.1, 0.5, 1.0, 2.0,5.0,10.0),))
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
        # output_size=7,
        # sampling_ratio=2)
        # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
        #     output_size=14,
        #     sampling_ratio=2)


        # model = MaskRCNN(
        #     backbone, 
        #     num_classes=6, 
        #     # min_size=512,
        #     rpn_anchor_generator=anchor_generator,
        #     box_roi_pool=roi_pooler,
        #     mask_roi_pool=mask_roi_pooler
        # )

        # model = FasterRCNN(
        #     backbone, 
        #     num_classes=6, 
        #     # min_size=512,
        #     rpn_anchor_generator=anchor_generator,
        #     box_roi_pool=roi_pooler,
        #     # mask_roi_pool=mask_roi_pooler
        # )


        if os.path.exists("./weight/model.pt"):
            if torch.cuda.is_available():
                model.load_state_dict(torch.load("./weight/model.pt"), strict=True)
            else:
                model.load_state_dict(torch.load("./weight/model.pt", map_location='cpu'), strict=True)
            
            print("="*6, "\nModel loaded, start retraining", "\n"+"="*6)
        else:
            print("="*6, "\nModel isn't found, train the network from begining.","\n"+"="*6)
        
        operator = op.Operator(model)
        operator.trainer(FLAGS.img_path, FLAGS.gt_path, FLAGS.bsize, FLAGS.lr, FLAGS.epoch)

    if FLAGS.predict:
        num_classes = 6

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 512
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        print(model)

        if os.path.exists("./weight/model.pt"):
            if torch.cuda.is_available():
                model.load_state_dict(torch.load("./weight/model.pt"), strict=True)
            else:
                model.load_state_dict(torch.load("./weight/model.pt", map_location='cpu'), strict=True)
            print("="*6, "\nModel loaded, start prediction", "\n"+"="*6)
        else:
            print("="*6, "\nModel isn't found, train the network first.","\n"+"="*6)
            sys.exit()

        with torch.no_grad():
            operator = op.Operator(model)
            operator.predictor(FLAGS.img_path)
