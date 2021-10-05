# coding=utf-8
"""Visualize the ActEV world coordinates."""

import argparse
import cv2
import os
import sys
import glob
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("h_path", help="path to all the homography matrix")
parser.add_argument("vis_path_world")
parser.add_argument("vis_path_img")


def plot_traj(img, traj, color):
  """Plot arrowred trajectory."""
  # traj is [N, 2]
  points = zip(traj[:-1], traj[1:])
  for p1, p2 in points:
    img = cv2.arrowedLine(img, tuple(p1), tuple(p2), color=color, thickness=1,
                          line_type=cv2.LINE_AA, tipLength=0.2)

  return img


def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


def world_to_img(w_xy, h_):
  """Transform image xy to world ground plane xy."""
  w_x, w_y = w_xy
  img_x, img_y, img_z = np.tensordot(
      h_, np.array([w_x, w_y, 1]), axes=1)
  return img_x / img_z, img_y / img_z


if __name__ == "__main__":
  args = parser.parse_args()

  delim = "\t"
  h, w = 1080, 1920

  if not os.path.exists(args.vis_path_img):
    os.makedirs(args.vis_path_img)
  if not os.path.exists(args.vis_path_world):
    os.makedirs(args.vis_path_world)

  h_files = glob.glob(os.path.join(args.h_path, "*.txt"))
  h_dict = {}  # image to world coordinates
  for h_file in h_files:
    scene = os.path.splitext(os.path.basename(h_file))[0]
    h_matrix = []
    with open(h_file, "r") as f:
      for line in f:
        h_matrix.append(line.strip().split(","))
    h_matrix = np.array(h_matrix, dtype="float")

    _, h_dict[scene] = cv2.invert(h_matrix)  # world to image

  traj_files = glob.glob(os.path.join(args.traj_path, "*.txt"))

  for traj_file in tqdm.tqdm(traj_files):
    videoname = os.path.splitext(os.path.basename(traj_file))[0]
    scene = get_scene(videoname)

    # all trajectory in one image
    vis_world = np.zeros((h, w, 3), dtype="uint8")
    vis_img = np.zeros((h, w, 3), dtype="uint8")

    data = []
    with open(traj_file, "r") as f:
      for line in f:
        data.append(line.strip().split(delim))
    # [frameIdx, personId, x, y]
    data = np.array(data, dtype="float32")

    # normalize the world coordinates to plot on h, w
    min_x = np.amin(np.array(data)[:, 2])
    max_x = np.amax(np.array(data)[:, 2])
    min_y = np.amin(np.array(data)[:, 3])
    max_y = np.amax(np.array(data)[:, 3])
    length_x = max_x - min_x
    length_y = max_y - min_y

    personIds = np.unique(data[:, 1]).tolist()

    for personId in personIds[:10]:

      all_points = data[data[:, 1] == personId, :]

      # sort according to frameidx
      all_points = all_points[np.argsort(all_points[:, 0]), :]

      all_points_world = all_points.copy()
      all_points_world[:, 2] = w * (all_points_world[:, 2] - min_x) / length_x
      all_points_world[:, 3] = h * (all_points_world[:, 3] - min_y) / length_y
      vis_world = plot_traj(vis_world, all_points_world[:, 2:], (0, 255, 0))

      all_points_img = np.array(
          [world_to_img(point, h_dict[scene]) for point in all_points[:, 2:]],
          dtype="float32")

      vis_img = plot_traj(vis_img, all_points_img, (0, 255, 0))

    target_file = os.path.join(args.vis_path_img, "%s.png" % (videoname))
    cv2.imwrite(target_file, vis_img)
    target_file = os.path.join(args.vis_path_world, "%s.png" % (videoname))
    cv2.imwrite(target_file, vis_world)










