# Chart Information Extraction

*For now, it's personal use, since not everything are ready*

## Data Preprocessing

#### Processing with Padding
- Original image file in : images/
- Original gt file in: json_gt/

According to sample_list.json:

```python3
python3 data_process/sampling.py
```

Generate the sampled images and json_gt in:
- images_sampled/
- json_gt_sampled/

Randomly padding images and modify the gt according to padding strategy:

```python3
python3 data_process/padding.py
```

Generate the padded images and json_gt in:
- padded_images_sampled/
- padded_json_gt_sampled/

Resize all images into 512*512 by running:

```python3
python3 data_process/img_gt_resize.py
```
Generate the resized and padded images an json_gt in:
- rs_padded_images_sampled/
- rs_padded_json_gt_sampled/

**This img_gt_resize.py code would reverse the horizontal chart's x and y axis in json_gt files, while original horizontal chart's x is vertical axis and y axis is horizontal axis**

According to images and json gt showed in the directory right above, generate the ground truth masks for the model, where masks contain ground truth of both classicification and regression.

Run the following code to split the images into train, val and test:
``` 
python3 data_process/split_train_val.py
```

where train, eval and test images would be within:
- rs_padded_images_sampled/train/
- rs_padded_images_sampled/eval/
- rs_padded_images_sampled/test/


Then Datasets are ready for the model usage

#### Processing without Padding

If no padding during processing, just ignore the python3 data_process/padding.py step above and rip off "padded_" in name of each directory.

Change the path name by the same rule indicated above within each python3 code at the begining where define the path.

## Model

Refer to the readme.md in Pytorch_UNet_ticks_C_R dir