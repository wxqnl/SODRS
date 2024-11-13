
# Semi-Supervised Learning for One-Stage Small Object Detection in Remote Sensing Images(SODRS)

## Synopsis
SODRS is implemented using the paddlepaddle framework and uses FCOS, which incorporates the student teacher model, as a baseline
## Precision Reference
The dataset is NWPUVHR-10
### [SODRS](configs/semi_det/sodrs)

|   Model    | Proportion of Surveillance Data | Semi mAP<sup>val<br>0.5:0.95 |
|:----------:|:-------------------------------:|:----------------------------:|
| SODRS-FCOS |               10%               |          **36.59**           |

### [FCOS](focs)

|   Model    | Proportion of Surveillance Data | Semi mAP<sup>val<br>0.5:0.95 |
|:----------:|:-------------------------------:|:----------------------------:|
| FCOS |               10%               |          **25.63**           |


## Semi-supervised dataset preparation

Semi-supervised object detection** requires both labelled and unlabelled data**, and the amount of unlabelled data is generally** much more than the amount of labelled data**.
There are generally two conventional settings for COCO tupe datasets:

(1) A partial proportion of the original training set `train` is extracted as labelled and unlabelled data;

Extracted from `train` at a fixed percentage (1%, 2%, 5%, 10%, etc.), and since the extraction method can have a large impact on the results of semi-supervised training, it was evaluated using 50% discount cross validation. 
The script produced by running the dataset division is as follows:
```bash
python tools/gen_semi_coco.py
```
The `train` full set will be divided according to the proportion of supervised data of 1%, 2%, 5%, and 10%, and each division will be randomly repeated 5 times for cross validation, and the generated semi-supervised labelled file is as follows:
- Labelled dataset annotation: `instances_train.{fold}@{percent}.json`
- Unlabelled dataset labelling: `instances_train.{fold}@{percent}-unlabeled.json`.
where `fold` denotes cross-validation and `percent` denotes the percentage of labelled data.
### Train

```bash
# Single-card training (need to adjust the learning rate accordingly on a linear scale)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/sodrs/sodrs_fcos_r50_fpn_coco_semi010.yml --eval

# Doka training
python -m paddle.distributed.launch --log_dir=sodrs_fcos_semi010/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/semi_det/sodrs/sodrs_fcos_r50_fpn_coco_semi010.yml --eval
```

### Valuation

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/semi_det/sodrs/sodrs_fcos_r50_fpn_coco_semi010.yml -o weights=output/sodrs_fcos_r50_fpn_coco_semi010/model_final.pdparams
```

### Anticipate

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/semi_det/sodrs/sodrs_fcos_r50_fpn_coco_semi010.yml -o weights=output/sodrs_fcos_r50_fpn_coco_semi010/model_final.pdparams --infer_img=..
```

### Deployments


```bash
# Export model
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/semi_det/sodrs/sodrs_fcos_r50_fpn_coco_semi010.yml -o weights=output/sodrs_fcos_r50_fpn_coco_semi010/model_final.pdparams
```

## Quote

```

```
