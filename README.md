# InternLSTM
In this work, we introduce InternLSTM, a framework based on the features extracted by InternVL.

***

## Main Results
| Model |  Ego4D (mAP)  |
| :-----: | :---: | 
|  GazePose  |   76.29 |
|  InternLSTM  |   76.57 |
|  InternLSTM*  |   76.94 |
|  InternLSTM**  |   77.18 |
***

Here, InternLSTM refers to the results obtained by training on the original training dataset and testing on the original test dataset. InternLSTM$* refers to the results obtained by training on a dataset combining the original training dataset and the flipped training dataset, and testing on the original test dataset. InternLSTM** refers to the average results obtained by training on a dataset combining the original training set and the flipped training set, and testing on both the original test dataset and the flipped test dataset.

## Quick Start

### Data preparation
You should prepare your data just as [*GazePose*](https://github.com/lemon-prog123/GazePose).
Specially, your structure should be like this:

```
Ego4D_LookAtMe/
* csv/
* json/
* split/
* result_LAM/
* json_original
* video_imgs/
* face_imgs/
  * uid
    * trackid
      * face_xxxxx.jpg
* videos_challenge/
  * video_id/
    * track_id/
      * unique_id.jpg
```
In addition, it is necessary to use [InternVL](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) to extract the features of each original image and the inverted image in the dataset. You can use [internfea.py](internfea.py) to get the features. Training dataset and validation dataset will be placed under internvitfea folder, and test dataset will be placed under internvitfea_test folder, both of them have the same architecture with face_imgs and videos_challenge. We generate 4 folders to place the features, internvitfea, internvitfea_flip, internvitfea_test, internvitfea_test_flip.

### 2. Train
Run the following script to start training for InternLSTM:
```
python run.py
```
Specify the arguments listed in [common/config.py](./common/config.py) if you want to customize the training.

Run the following script to start training for InternLSTM*:
```
python run_mul_traindataset.py
```


### 3. Inference

Run the following script for inference InternLSTM:
```
python run.py --eval --checkpoint ${checkpoint_path} --exp_path ${eval_output_dir}
```
Run the following script for inference InternLSTM*:
```
python run_mul_traindataset.py --eval --checkpoint ${checkpoint_path} --exp_path ${eval_output_dir}
```
If you want to inference InternLSTM**, you need to run like InternLSTM*:
```
python run_mul_traindataset.py --eval --checkpoint ${checkpoint_path} --exp_path ${eval_output_dir} --flip_test
```
Then you need to use [catcsv.py](catcsv.py) to calculate the average result of the test dataset before and after flip.
