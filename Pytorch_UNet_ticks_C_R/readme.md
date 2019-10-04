### Train
python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_masks_sampled/" --lr=0.0001 --epoch=15 --bsize=16

python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_masks_sampled/" --lr=0.0001 --epoch=24 --bsize=20

python3 main.py --train --img_path="../../data/SUMIT/rs_images_sampled/" --gt_path="../../data/SUMIT/rs_masks_sampled/" --lr=0.0001 --epoch=15 --bsize=16

python3 main.py --train --img_path="../../data/SUMIT/rs_padded_images_sampled/" --gt_path="../../data/SUMIT/rs_padded_masks_sampled/" --lr=0.0001 --epoch=24 --bsize=20

### Predict 
python3 main.py --predict --img_path="/home/csgrad/pyan4/data/PMC/images/tasks345/" 

python3 main.py --predict --img_path="/home/csgrad/pyan4/data/SUMIT/rs_images_sampled/test/"

python3 main.py --predict --img_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_images_sampled/test/"

python3 main.py --predict --vis --img_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_images_sampled/test/"

python3 main.py --predict --img_path="/home/csgrad/pyan4/data/SUMIT/task345_test/images/"

python3 main.py --predict --vis --img_path="/home/csgrad/pyan4/data/SUMIT/task345_test/images/"

python3 main.py --predict --img_path="../../data/SUMIT/rs_images_sampled/test/" 

python3 main.py --predict --img_path="../../data/PMC/images/tasks345/" 

### For nohup
rm -rf nohup.out && nohup python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_masks_sampled/" --lr=0.0001 --epoch=15 --bsize=16

rm -rf nohup.out && nohup python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_padded_masks_sampled/" --lr=0.0001 --epoch=15 --bsize=16