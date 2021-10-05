# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given the person boxes, extract CNN features from the boxes."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import argparse
import cv2
import itertools
import os
import numpy as np
import cPickle as pickle
import tensorflow as tf
import sys

from tqdm import tqdm
from glob import glob

from utils import resize_image

# assuming you run this script at the script's path
# $ git clone https://github.com/JunweiLiang/Object_Detection_Tracking
# $ cd  Object_Detection_Tracking
# $ git checkout 04f24336a02efb7671760a5e1bacd27141a8617c
sys.path.append("Object_Detection_Tracking")
from nn import resnet_fpn_backbone
from nn import fpn_model

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("person_box_path")

parser.add_argument("frame_path", help="path to video frames")
parser.add_argument("modelpath")

parser.add_argument("out_path", help="path to save train/ val/ test/boxid.npy"
                                     " for the person box feature")
parser.add_argument("personid_to_boxid_mapping",
                    help="personid[framenum_trackid] to boxid mapping"
                         " for each split")

# so here we assume the boxes are under 1920x1080 scale.
parser.add_argument("--imgh", default=1080, type=int)
parser.add_argument("--imgw", default=1920, type=int)
parser.add_argument("--person_h", default=9, type=int)
parser.add_argument("--person_w", default=5, type=int)

parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--runfold", default="train|val|test")


# pylint: disable=g-line-too-long
def roi_align(featuremap, boxes, output_shape_h, output_shape_w):
  """Modified roi_align to allow for non-rectangle output shape. Origin: https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_box.py ."""
  boxes = tf.stop_gradient(boxes)
  box_ind = tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32)

  image_shape = tf.shape(featuremap)[2:]
  crop_shape = [output_shape_h * 2, output_shape_w * 2]
  x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

  spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
  spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

  nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
  ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

  nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(
      image_shape[1] - 1)
  nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(
      image_shape[0] - 1)

  boxes = tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

  featuremap = tf.transpose(featuremap, [0, 2, 3, 1])
  ret = tf.image.crop_and_resize(
      featuremap, boxes, box_ind,
      crop_size=[crop_shape[0], crop_shape[1]])
  ret = tf.transpose(ret, [0, 3, 1, 2])

  ret = tf.nn.avg_pool(ret, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
                       padding="SAME", data_format="NCHW")
  return ret


# pylint: disable=invalid-name
class ModelFPN(object):
  """FPN backbone model for extracting features given boxes."""

  def __init__(self, config):
    self.config = config
    H = self.imgh = config.imgh
    W = self.imgw = config.imgw
    # image input, one image per person box
    # [H,W,3]
    self.imgs = tf.placeholder("float32", [1, H, W, 3], name="img")
    # [4]
    self.boxes = tf.placeholder("float32", [1, 4], name="person_box")

    # the model graph
    with tf.name_scope("image_prepro"):
      images = self.imgs
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      # cv2 load image is bgr
      mean = mean[::-1]
      std = std[::-1]
      image_mean = tf.constant(mean, dtype=tf.float32)
      image_std = tf.constant(std, dtype=tf.float32)

      images = images*(1.0/255)
      images = (images - image_mean) / image_std
      images = tf.transpose(images, [0, 3, 1, 2])

    with tf.name_scope("fpn_backbone"):
      c2345 = resnet_fpn_backbone(images, config.resnet_num_block,
                                  resolution_requirement=32,
                                  tf_pad_reverse=True,
                                  use_dilations=True)
      p23456 = fpn_model(c2345, num_channel=config.fpn_num_channel, scope="fpn")

    with tf.name_scope("person_box_features"):
      # NxCx7x7 # (?, 256, 7, 7)
      person_features = self.multilevel_roi_align(p23456[:4], self.boxes)

      # [1, 9, 5, 2048]
      self.person_features = tf.transpose(person_features, perm=[0, 2, 3, 1])

  def multilevel_roi_align(self, features, rcnn_boxes):
    """ROI align pooling feature from the right level of feature."""
    config = self.config
    assert len(features) == 4
    # Reassign rcnn_boxes to levels # based on box area size
    level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i_, boxes, featuremap in zip(itertools.count(), level_boxes, features):
      with tf.name_scope("roi_level%s" % (i_ + 2)):
        boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i_])
        all_rois.append(roi_align(featuremap, boxes_on_featuremap,
                                  config.person_h, config.person_w))

    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois

  def fpn_map_rois_to_levels(self, boxes):
    """Map rois to feature level based on box size."""
    def tf_area(boxes):
      x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
      return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.to_int32(tf.floor(4 + tf.log(
        sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))

    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]

    level_ids = [tf.reshape(x, [-1], name="roi_level%s_id" % (i_ + 2))
                 for i_, x in enumerate(level_ids)]

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes

  def get_feed_dict(self, imgfile, box):
    """Get feed dict to feed tf."""
    feed_dict = {}
    H = self.imgh
    W = self.imgw

    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    assert img is not None, imgfile
    img = img.astype("float32")

    resized_image = resize_image(img, H, W)

    newh, neww = resized_image.shape[:2]

    # comment the following if your image is not 16:9 scale
    assert newh == H, (newh, H)
    assert neww == W, (neww, W)

    # assuming box is already under H,W size image

    feed_dict[self.imgs] = resized_image.reshape(1, H, W, 3)

    feed_dict[self.boxes] = np.array(box).reshape(1, 4)

    return feed_dict


def process(config, split, sess):
  """Extract feature per split."""

  traj_path = os.path.join(config.traj_path, split)
  person_box_path = os.path.join(config.person_box_path, split)

  # we get person box based on trajectory files
  # since not all box in person_box_path are used
  traj_files = glob(os.path.join(traj_path, "*.txt"))

  out_path = os.path.join(config.out_path, split)

  if not os.path.exists(out_path):
    os.makedirs(out_path)

  # videoname_framenum_personid => boxid.npy; for preprocessing
  config.person_boxkey2boxid[split] = {}

  boxid = 0
  for traj_file in tqdm(traj_files):
    videoname = os.path.splitext(os.path.basename(traj_file))[0]

    # load the person box data
    with open(os.path.join(person_box_path, "%s.p" % videoname)) as f:
      person_box_data = pickle.load(f)  # framenum_personid => box

    # each line of the trajectory file should have one feature
    with open(traj_file, "r") as f:
      for line in f:
        framenum, personid, _, _ = line.strip().split("\t")
        framenum, personid = int(float(framenum)), int(float(personid))  # yikes

        person_box = person_box_data["%d_%d" % (framenum, personid)]  # [4]
        framefile = os.path.join(config.frame_path, videoname,
                                 "%s_F_%08d.jpg"%(videoname, framenum))

        if not os.path.exists(framefile):
          print("warning, %s not exists, ignored" % (framefile))
          continue

        person_feature, = sess.run([
            config.model.person_features], feed_dict=config.model.get_feed_dict(
                framefile, person_box))

        np.save(os.path.join(out_path, "%s.npy" % boxid), person_feature)

        box_key = "%s_%d_%d" % (videoname, framenum, personid)
        config.person_boxkey2boxid[split][box_key] = boxid
        boxid += 1


def initialize(config, sess):
  """Load model weights into tf Graph."""
  tf.global_variables_initializer().run()
  print("restoring model...")
  allvars = tf.global_variables()
  allvars = [var for var in allvars if "global_step" not in var.name]
  restore_vars = allvars
  opts = ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
          "Adadelta", "Momentum"]
  restore_vars = [var for var in restore_vars
                  if var.name.split(":")[0].split("/")[-1] not in opts]

  saver = tf.train.Saver(restore_vars, max_to_keep=5)

  load_from = config.modelpath

  ckpt = tf.train.get_checkpoint_state(load_from)
  if ckpt and ckpt.model_checkpoint_path:
    loadpath = ckpt.model_checkpoint_path
    saver.restore(sess, loadpath)
    print("\tloaded %s" % loadpath)

  else:
    raise Exception("Model not exists")

if __name__ == "__main__":
  args = parser.parse_args()

  args.resnet_num_block = [3, 4, 23, 3]  # resnet 101

  args.fpn_num_channel = 256

  args.anchor_strides = (4, 8, 16, 32, 64)

  args.runfolds = args.runfold.split("|")

  args.model = ModelFPN(args)

  args.person_boxkey2boxid = {}

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [args.gpuid]]))

  with tf.Session(config=tfconfig) as session:
    initialize(args, session)
    for fold in args.runfolds:
      process(args, fold, session)

  with open(args.personid_to_boxid_mapping, "w") as fw:
    pickle.dump(args.person_boxkey2boxid, fw)
