## Prepare Rich Visual Features
These steps will allow you to generate the prepared data in the format as [here](https://github.com/google/next-prediction/blob/master/NOTES.md) and to conduct experiments starting from raw ActEV videos.

# Dependencies
Tested on TF 1.10. We will also need to use other repos: [this](https://github.com/JunweiLiang/Object_Detection_Tracking) and [this](https://github.com/JunweiLiang/deep-high-resolution-net.pytorch).

# Step 1. Get the videos
First, download all videos from the [official website](https://actev.nist.gov/) and put all videos into a folder named `actev_all_videos`. There should be 119 mp4 videos under this folder.
Example:
```
# All the commands below assumes to be under prepare_data/ folder.
$ cd code/prepare_data/

# Get the data split lists.
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/data_splits.tgz
$ tar -zxvf data_splits.tgz

# Extract video frames using opencv. Note all frames will be resized to 1920x1080
$ python step1_get_frames.py actev_all_videos/ actev_all_video_frames \
--resize --size 1080 --maxsize 1920
```

# Step 2. Get object/trajectory/activity ground truth labels
```
# Download the original ActEV labels. Please do get permission from the official website
$ wget https://next.cs.cmu.edu/data/actev-v1-drop4-yaml.tgz
$ tar -zxvf actev-v1-drop4-yaml.tgz
$ mkdir actev_all_annotations/
$ find actev-v1-drop4-yaml/ -name "*.yml" | while read line;do \
mv $line actev_all_annotations/; done

# Pack the ground truth tracks into a file for each video. Note here we rescale
# all bounding box to be under 1920x1080 scale.
# For real-world system, you could replace this part with outputs from
# a object detection & tracking system.
$ python step2_object_act_annotations.py data_splits/all.lst \
actev_all_annotations/ actev_all_obj-track-act
```

# Step 3. Get scene semantic segmentation features [ADE20K pre-trained model]
```
# Download the Deeplabv3 pre-trained model: deeplabv3_xception_ade20k_train
# Official website: https://github.com/tensorflow/models/tree/master/research/deeplab
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/deeplabv3_xception_ade20k_train.pb

# Get an ordered list of frames.
$ find $PWD/actev_all_video_frames/ -name "*.jpg" -print0 |sort -z| \
xargs -r0 echo|sed 's/ /\n/g'  > actev_all_video_frames.ordered.lst

# We skip some frames and downsize the features.
$ python step3_scene_semantics.py actev_all_video_frames.ordered.lst \
deeplabv3_xception_ade20k_train.pb actev_all_video_frames_scene_seg_every30_36x64 \
--every 30 --down_rate 8.0
```
[Here](https://next.cs.cmu.edu/data/072019_prepare_data/VIRAT_S_000007_ade20k.mp4) is a visualization of the output.

# Step 4. Generate trajectory data and all the runtime annotations
```
$ python step4_generate_traj.py actev_all_obj-track-act/ data_splits/ traj_2.5fps \
--drop_frame 12 --scene_feat_path actev_all_video_frames_scene_seg_every30_36x64/ \
--scene_map_path anno_scene --person_box_path anno_person_box --other_box_path \
anno_other_box --activity_path anno_activity
```

# Step 5. Get person appearance features
```
# Get the pre-trained object detection model to extract features.
# step 5 script will import necessary code from the repo.
$ git clone https://github.com/JunweiLiang/Object_Detection_Tracking
$ cd Object_Detection_Tracking/
$ git checkout 04f24336a02efb7671760a5e1bacd27141a8617c
$ wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/models/obj_v3_model.tgz
$ tar -zxvf obj_v3_model.tgz
$ cd ../

# Install all the dependencies according to the repo's README.

# Extract features given the person bounding boxes.
# Save a npy file per person per frame and also a mapping file.
$ python step5_person_appearance.py traj_2.5fps/ anno_person_box/ \
actev_all_video_frames Object_Detection_Tracking/obj_v3_model/ \
person_appearance_features person_boxkey2id.p --imgh 1080 --imgw 1920 \
--person_h 9 --person_w 5 --gpuid 0
```

# Step 6. Get person keypoint features
Different from in the paper, we switch to use [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) here. We have modified the original code (minimally) to allow inferencing on ActEV video frames and not exiting when an image file does not exist.
```
# Get my fork of the HRNet.
$ git clone https://github.com/JunweiLiang/deep-high-resolution-net.pytorch

# Install all the dependencies according to the repo's README.

# Get the COCO pre-trained model (pose_hrnet_w48_384x288.pth) by the HRNet authors.
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/pose_hrnet_w48_384x288.pth
# Also this thing
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/person_keypoints_val2017.json

# Convert the person bbox to json format for HRNet.
$ mkdir anno_person_box_json/
$ ls anno_person_box/*/*.p | while read line;do videoname=$(basename $line .p); \
python step6-1_get_person_json_box.py $line anno_person_box_json/${videoname}.json;done

# Run HRNet inferencing given person bounding boxes for each video.
$ mkdir hrnet_kp_out/
$ ls anno_person_box/*/* |while read line;do videoname=$(basename $line .p); \
python3 deep-high-resolution-net.pytorch/tools/test.py --cfg \
deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
TEST.MODEL_FILE pose_hrnet_w48_384x288.pth OUTPUT_DIR hrnet_kp_out/${videoname} \
VIDEONAME ${videoname} FRAMEPATH actev_all_video_frames/ TEST.USE_GT_BBOX False \
COCO_JSON person_keypoints_val2017.json TEST.COCO_BBOX_FILE \
anno_person_box_json/${videoname}.json TEST.BATCH_SIZE_PER_GPU 32 \
GPUS "(0,)" CHECK_IMG True;done

# Convert the HRNet output back to pickle files. You have to do this for 3 splits.
# You will see some non-zero fail rate, which is fine. This is due to the fact that
# the ActEV labels have some person bounding box in frames that do not exist.
$ split=val; mkdir -p anno_kp/${split}; ls $PWD/anno_person_box/${split}/* \
|while read line;do videoname=$(basename $line .p); python step6-2_hrnet_output_to_pickle.py \
$line hrnet_kp_out/${videoname} anno_kp/${split}/${videoname}.p;done

# Check for fail rate==1.0 videos. Run them again to fix them.
```
[Here](https://next.cs.cmu.edu/data/072019_prepare_data/VIRAT_S_040103_08_001475_001512_hrnet.mp4) is the person keypoint visualization.

# Preprocess
Now all the ingredients are ready. Note that the preprocess.py in this repo is slightly different the Google repo. This is [our prepared data (5.0 GB)](https://next.cs.cmu.edu/data/072019_prepare_data/072019_next_my_prepare.tgz) from these steps. The rest is similar to the steps in previous [training](https://github.com/google/next-prediction/blob/master/TRAINING.md) and [testing](https://github.com/google/next-prediction/blob/master/TESTING.md).
Example:
```
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/scene36_64_id2name_top10.json

# Preprocess
$ python ../preprocess.py traj_2.5fps/ actev_preprocess --obs_len 8 --pred_len 12 \
--add_kp  --kp_path anno_kp/ --add_scene  --scene_feat_path \
actev_all_video_frames_scene_seg_every30_36x64/ --scene_map_path anno_scene/ \
--scene_id2name scene36_64_id2name_top10.json --scene_h 36 --scene_w 64 \
--video_h 1080 --video_w 1920 --add_grid --add_person_box --person_box_path \
anno_person_box/ --add_other_box --other_box_path anno_other_box/ --add_activity \
--activity_path anno_activity/ --person_boxkey2id_p person_boxkey2id.p
# There will be several warnings. In our case it is 6 for training set.

```

# Train
```
$ python ../train.py actev_preprocess/ next-models/actev_single_model model \
--runId 0 --is_actev --add_kp --add_activity --person_feat_path \
person_appearance_features/ --multi_decoder --batch_size 64
```

# Test
```
$ python ../test.py actev_preprocess/ next-models/actev_single_model model \
--runId 0 --is_actev --add_kp --add_activity --person_feat_path \
person_appearance_features/ --multi_decoder --batch_size 64 --load_best
```
[Our model](https://next.cs.cmu.edu/data/072019_prepare_data/next-models.tgz) got the following results. The small difference from the ones reported in the paper may be due to different keypoint features and appearance features.
<table>
  <tr>
    <td>Activity mAP</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.199</td>
    <td>18.11</td>
    <td>37.51</td>
  </tr>
</table>

# Run Inference on New Videos
To run the trajectory/activity prediction models trained on ActEV on other video datasets, replace Step 2 with the output from a object detection & tracking system.
