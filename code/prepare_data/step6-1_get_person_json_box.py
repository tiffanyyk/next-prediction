# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Get person box in HRnet json format."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import argparse
import cPickle as pickle
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("box_pickle")
parser.add_argument("box_json")


def convert_box(box):
  """Convert box to coco format."""
  x1, y1, x2, y2 = box
  return [x1, y1, x2 - x1, y2 - y1]


if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.box_pickle, "r") as f:
    box_dict = pickle.load(f)

  videoname = os.path.splitext(os.path.basename(args.box_pickle))[0]

  boxes = []
  for framenum_pid in box_dict:
    framenum, pid = framenum_pid.split("_")
    framenum = int(framenum)
    pid = int(pid)

    boxes.append({
        "image_id": framenum,
        "pid": pid,
        "category_id": 1,
        "bbox": convert_box(box_dict[framenum_pid]),
        "score": 1.0
    })

  with open(args.box_json, "w") as f:
    json.dump(boxes, f)
