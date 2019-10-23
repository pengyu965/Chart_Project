import os 
import cv2 
import torch 
from torchvision import datasets, transforms
from tqdm import tqdm

train_data_path = "../data/MIX/original/"

transformer = transforms.ToTensor()

mean = 0.0 
sumvar = 0.0
count = 0
for image in tqdm(os.listdir(train_data_path)):
    img = cv2.imread(train_data_path+image)
    img_array = transformer(img)

    mean += img_array.mean([1,2])
    
mean = mean/len(os.listdir(train_data_path))
print("Mean:", mean)

for image in tqdm(os.listdir(train_data_path)):
    img = cv2.imread(train_data_path+image)
    img_array = transformer(img)

    var = (img_array-mean.unsqueeze(1).unsqueeze(1))**2.

    sumvar += var.sum([1,2])
    count += torch.numel(img_array[0])

std = torch.sqrt(sumvar/count)
print("Std:", std)

with open("../data/mean_std.json",'w') as f:
    f.write("MIX dataset: Mean is {}, Std is {}".format(mean, std))