### Train
python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_masks_sampled/" --lr=0.0001 --epoch=15 --bsize=16

### Predict 
python3 main.py --predict --img_path="/home/csgrad/pyan4/data/PMC/images/tasks345/" 