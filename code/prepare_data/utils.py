# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Utils."""

import cv2

# These are the activities we care
activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}

object2id = {
    "Person": 0,
    "Vehicle": 1,
    "Parking_Meter": 2,
    "Construction_Barrier": 3,
    "Door": 4,
    "Push_Pulled_Object": 5,
    "Construction_Vehicle": 6,
    "Prop": 7,
    "Bike": 8,
    "Dumpster": 9,
}

actev_scene2imgsize = {
    "0002": (1280.0, 720.0),
    "0000": (1920.0, 1080.0),
    "0400": (1920.0, 1080.0),
    "0401": (1920.0, 1080.0),
    "0500": (1920.0, 1080.0),
}


def get_scene(videoname):
  """ActEV scene extractor from videoname."""
  s = videoname.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


# actev boxes may contain some errors
# won't fix x,y reversed
def modify_box(bbox, imgsize):
  """Modify ActEV boxes."""
  w, h = imgsize
  x1, y1, x2, y2 = bbox
  x_min = min(x1, x2)
  x_max = max(x1, x2)
  y_min = min(y1, y2)
  y_max = max(y1, y2)

  x_min = min(w, x_min)
  x_max = min(w, x_max)
  y_min = min(h, y_min)
  y_max = min(h, y_max)

  return [x_min, y_min, x_max, y_max]


def valid_box(box, wh):
  """Check whether boxes are within the image bounds."""
  w, h = wh
  a = box_area(box)
  if a <= 0:
    return False
  if (box[0] > w) or (box[2] > w) or (box[1] > h) or (box[3] > h):
    return False
  return True


def box_area(box):
  """compute bbox area size in pixels."""
  x1, y1, x2, y2 = box
  w = x2 - x1
  h = y2 - y1
  return float(w) * h


# given x1,y1,x2,y2 box,
# get the feet location.
def get_traj_point(box):
  """Get person traj point given person boxes."""
  x1, _, x2, y2 = box
  return [(x1 + x2) / 2.0, y2]


def resize_image(im, short_size, max_size):
  """Duh."""
  h, w = im.shape[:2]
  neww, newh = get_new_hw(h, w, short_size, max_size)
  return cv2.resize(im, (neww, newh), interpolation=cv2.INTER_LINEAR)


def get_new_hw(h, w, size, max_size):
  """Get the new h,w, keeping original ratio."""
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
