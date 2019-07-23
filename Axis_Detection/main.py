import os 
import torch 
import argparse
import sys
import cv2
import json
import numpy as np
import torchvision
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import utils
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


parser = argparse.ArgumentParser()

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

cfg = parser.parse_args()


class axisdataset(Dataset):
    def __init__(self, img_root, gt_root, transforms):
        self.img_root = img_root
        self.gt_root = gt_root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.image_list = os.listdir(self.img_root)
        

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_root, self.image_list[idx])
        gt_path = os.path.join(self.gt_root, self.image_list[idx][:-4]+".json")
        gt = json.load(open(gt_path, 'r'))
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        boxes = []
        for axis in gt["input"]["task4_output"]["axes"]:
            for item in gt["input"]["task4_output"]["axes"][axis]:
                x = item["tick_pt"]["x"]
                y = item["tick_pt"]["y"]
                x0 = x - 5
                y0 = y - 5
                x1 = x + 5
                y1 = y + 5
                boxes.append([x0,y0,x1,y1])
        
        num_objs = len(boxes)
        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # print(boxes.shape)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return (img, target)

    def __len__(self):
        return len(self.image_list)

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((3, 5, 10, 15),),
                                aspect_ratios=((0.5, 1.0, 2),))


roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                            output_size=7,
                                            sampling_ratio=2)

model = FasterRCNN(backbone,
                num_classes=2,
                min_size = 512,
                # max_size = 1400,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler)

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

if cfg.train:

    dataset = axisdataset("../../data/SUMIT/rs_images_sampled/", "../../data/SUMIT/rs_json_gt_sampled/", None)

    dataloader = DataLoader(dataset = dataset, batch_size = 2, num_workers = 28, collate_fn=utils.collate_fn)

    model.to(device)

        # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 0.005
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    epoch = 10
    for ep in range(epoch):
        if ep == int(epoch //3):
            lr = lr/10
            optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
        if ep == int(epoch*2//3):
            lr = lr/10
            optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)

        for idi, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print("===Epoch:{}/{}===Step:{}/{}===Loss:{:.4f}".format(ep, epoch, idi, len(dataloader), losses.item()))
            # print(loss_dict)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # model.eval()
            # prediction = model(images)
            # print(prediction[0])

        torch.save(model.state_dict(), "./weight_2/{}.pt".format(ep))
    # img_array = predictions.permute(1,2,0).detach().cpu().numpy().astype(uint8)
    # cv2.imshow("img", cv2.fromarray(img_array))

if cfg.predict:
    img_path = "../../data/SUMIT/rs_images_sampled/"
    dataset = os.listdir(img_path)
    indices = torch.randperm(len(dataset)).tolist()

    
    model.load_state_dict(torch.load("./weight/9.pt"), strict = False)
    model.to(device)
    model.eval()
    
    for idi in indices[-10:]:
        img = Image.open(img_path+dataset[idi]).convert("RGB")
        # print(np.array(img).shape)
        img = torch.tensor(np.array(img)).float().permute(2,0,1).unsqueeze(0).to(device)
        predict = model(img)
        boxes_list = predict[0]["boxes"].data.cpu().numpy()
        print(predict[0]["boxes"].data.cpu().numpy())

        iimg = cv2.imread(img_path+dataset[idi])
        for box in boxes_list[0]:
            cv2.rectangle(iimg, (box[0],box[1]), (box[2], box[3]), (0,0,255), 2)
        
        cv2.imwrite("./samples/"+dataset[idi])

    

