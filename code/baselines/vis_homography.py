# coding=utf-8
"""Given image and homography matrix, visualize the homograph."""

from __future__ import print_function

import argparse
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("homography")
parser.add_argument("new_image")


def read_h(h_file):
  """Read the homography matrix into np."""
  h_matrix = np.zeros((3, 3), dtype="float")
  for i, line in enumerate(open(h_file, "r").readlines()):
    h_matrix[i, :] = [float(x) for x in line.strip().split(",")]

  return h_matrix


if __name__ == "__main__":
  args = parser.parse_args()

  image = cv2.imread(args.image)

  h, w, c = image.shape
  print(h, w)

  overlay = np.zeros((h, w, 3), dtype="uint8")

  H = read_h(args.homography)
  _, H_inv = cv2.invert(H)

  # [x1, y1, x2, y2]
  image_box = (0, 0), (h, 0), (0, w), (h, w)

  world_box = []
  for x, y in image_box:
    w_x, w_y, w_z = np.tensordot(H, np.array([x, y, 1]), axes=1)
    world_box.append((w_x/w_z, w_y/w_z))

  xy = np.float32(image_box).reshape(-1, 1, 2)
  world_box_xy = cv2.perspectiveTransform(
      xy, H)
  # these are the same
  print(world_box)
  print(world_box_xy)
  world_x1, world_y1 = world_box_xy[0, 0, :]
  world_x2, world_y2 = world_box_xy[-1, 0, :]
  world_x1, world_y1 = -10, -10
  world_x2, world_y2 = 100, 100

  step = 100
  step_size_x = (world_x2 - world_x1) / step
  step_size_y = (world_y2 - world_y1) / step
  for step_x in range(step):
    for step_y in range(step):
      this_world_x = world_x1 + step_x * step_size_x
      this_world_y = world_y1 + step_y * step_size_y
      image_x, image_y, z = np.tensordot(
          H_inv, np.array([this_world_x, this_world_y, 1]), axes=1)
      image_x /= z
      image_y /= z
      # image_xy = cv2.perspectiveTransform(np.array([[[this_world_x, this_world_y]]]), H_inv)
      # image_x, image_y = np.squeeze(image_xy)

      if (image_x >= 0) and (image_x < w) and (image_y >= 0) and (image_y < h):
        overlay[int(image_y), int(image_x), 2] = 255

  new_image = cv2.addWeighted(image, 1.0, overlay, 1.0, 0)

  cv2.imwrite(args.new_image, new_image)
