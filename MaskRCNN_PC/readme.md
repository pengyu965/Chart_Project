### Train
python3 main.py --train --img_path="/home/csgrad/pyan4/data/SUMIT/rs_images_sampled/" --gt_path="/home/csgrad/pyan4/data/SUMIT/rs_json_gt_sampled/" --lr=0.0001 --epoch=24 --bsize=2

python3 main.py --train --img_path="../../data/SUMIT/rs_images_sampled/" --gt_path="../../data/SUMIT/pc_json_gt_sampled/" --lr=0.0001 --epoch=24 --bsize=2

python3 main.py --train --img_path="../../data/SUMIT/padded_images_sampled/" --gt_path="../../data/SUMIT/padded_json_gt_sampled/" --lr=0.0001 --epoch=24 --bsize=2


## Fine tune
python3 main.py --train --img_path="../../data/finetune/" --gt_path="../../data/finetune/gt/" --lr=0.0001 --epoch=24 --bsize=20

### Predict 
python3 main.py --predict --img_path="/home/csgrad/pyan4/data/PMC/tasks345_data/rs_images/" 

python3 main.py --predict --img_path="../../data/PMC/images/tasks345/" 

### For nohup
rm -rf nohup.out && nohup python3 main.py --train --img_path="../../data/SUMIT/rs_images_sampled/" --gt_path="../../data/SUMIT/rs_json_gt_sampled/" --lr=0.0001 --epoch=24 --bsize=2
