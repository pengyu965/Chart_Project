import os 
import torch 
import argparse
import sys

from model import googlenet_model
from model import op


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
    parser.add_argument("--data", type= str, help='input_data')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default = 10, 
                        help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=60,
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
    

    if os.path.exists("./weight/") == False:
        os.mkdir("./weight/")

    
    if os.path.exists("./weight/GoogLeNet/") == False:
            os.mkdir("./weight/GoogLeNet/")
        
    if FLAGS.train or FLAGS.finetune: 

        if FLAGS.finetune:
            model = googlenet_model.GoogLeNet(num_classes=FLAGS.class_num, aux_logits=False)

            # for param in model.parameters():
            #     param.requires_grad = False
            # # for param in model.aux1.parameters():
            # #     param.requires_grad = True
            # # for param in model.aux2.parameters():
            # #     param.requires_grad = True 
            # for param in model.fc.parameters():
            #     param.requires_grad = True

            try:
                model_state = torch.load("./weight/GoogLeNet/weight.pt", map_location = 'cpu')
                filtered_dic = {}
                for param_name, param_tensor in model_state.items():
                    if "fc" not in param_name and "aux" not in param_name:
                        # print(param_name)
                        filtered_dic[param_name] = param_tensor
                model.load_state_dict(filtered_dic, strict = False)
                print("\n***\nCheckpoint found\nPart Model Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found or Model State error\nFine-tune Stopped.\n***\n")
        
    
        else:
            model = googlenet_model.GoogLeNet(num_classes=FLAGS.class_num)

            try:
                model.load_state_dict(torch.load("./weight/GoogLeNet/weight.pt"), strict = False)
                print("\n***\nCheckpoint found\nModel Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found\nTraining from begining\n***\n")
        
        print(model)
        operator = op.Operator(model, FLAGS.bsize, FLAGS.lr, FLAGS.epoch, FLAGS.class_num)
        
        training_data = op.Chartdata(input_path = FLAGS.data+"/train.json")   
        val_data = op.Chartdata(input_path = FLAGS.data+"/val.json")

        operator.trainer(train_data = training_data, val_data = val_data)
    
    if FLAGS.eval:
        model = googlenet_model.GoogLeNet(num_classes=FLAGS.class_num)
        print(model)

        try:
            model.load_state_dict(torch.load("./weight/GoogLeNet/weight.pt"), strict = False)
            print("\n***\nCheckpoint found\nModel Restored\nBegin Evaluation\n***\n")
        except:
            print("\n***\nNo Checkpoint found\nTrain the Model First\n***\n")
            sys.exit()

        operator = op.Operator(model, FLAGS.bsize, FLAGS.lr, FLAGS.epoch, FLAGS.class_num)
        
        val_data = op.Chartdata(input_path = FLAGS.data+"/val.json")

        operator.validator(val_data = val_data, s_writer = False, verbose = True)

        




    