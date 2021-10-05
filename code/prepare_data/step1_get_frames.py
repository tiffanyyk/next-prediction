# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given video path, extract video frames and resize them."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import argparse
import cv2
import tqdm
import os
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("video_path")
parser.add_argument("frame_path")

parser.add_argument("--size", default=1080, type=int)
parser.add_argument("--maxsize", default=1920, type=int)

parser.add_argument("--resize", default=False, action="store_true")

# For running parallel jobs, set --job 4 --curJob k, where k=1/2/3/4
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="this script run job Num")


def get_new_hw(h, w, size, max_size):
  """Get the new img size with the same ratio."""

  scale = size * 1.0 / min(h, w)
  if h < w:
    newh, neww = size, scale * w
  else:
    newh, neww = scale * h, size
  if max(newh, neww) > max_size:
    scale = max_size * 1.0 / max(newh, neww)
    newh = newh * scale
    neww = neww * scale
  neww = int(neww + 0.5)
  newh = int(newh + 0.5)
  return neww, newh


if __name__ == "__main__":
  args = parser.parse_args()

  print("using opencv version:%s" % (cv2.__version__))

  video_files = glob(os.path.join(args.video_path, "*"))
  video_files.sort()

  count = 0
  for video_file in tqdm(video_files):
    count += 1
    if (count % args.job) != (args.curJob - 1):
      continue

    video_name = os.path.splitext(os.path.basename(video_file))[0]

    target_path = os.path.join(args.frame_path, video_name)
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    try:
      vcap = cv2.VideoCapture(video_file)
      if not vcap.isOpened():
        raise Exception("cannot open %s" % video_file)
    except Exception as e:
      raise e

    if cv2.__version__.split(".")[0] != "2":
      frame_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
      frame_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

      fps = vcap.get(cv2.CAP_PROP_FPS)
      frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
      frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
      frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

      fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
      frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    cur_frame = 0
    while cur_frame < frame_count:
      suc, frame = vcap.read()

      if not suc:
        cur_frame += 1
        tqdm.write("warning, %s frame of %s failed" % (cur_frame,
                                                       video_name))
        continue

      frame = frame.astype("float32")

      if args.resize:
        img_w, img_h = get_new_hw(frame.shape[0], frame.shape[1],
                                  args.size, args.maxsize)

        frame = cv2.resize(frame, (img_w, img_h),
                           interpolation=cv2.INTER_LINEAR)

      cv2.imwrite(os.path.join(target_path,
                               "%s_F_%08d.jpg" % (video_name, cur_frame)),
                  frame)

      cur_frame += 1






