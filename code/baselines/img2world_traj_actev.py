# coding=utf-8
"""Given the traj path, make new path with homography transformed and keeping
the data split.
"""

import argparse
import glob
import os
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("h_path",
                    help="path to homography matrices for each scene.")
parser.add_argument("new_traj_path")
parser.add_argument("--translate_positive", action="store_true",
                    help="translate the world coordinates so they are all"
                    " positive, but then the H matrix will break [not useful]")


def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


def get_world_coordinates(img_xy, h_):
  """Transform image xy to world ground plane xy."""
  img_x, img_y = img_xy
  world_x, world_y, world_z = np.tensordot(
      h_, np.array([img_x, img_y, 1]), axes=1)
  return world_x / world_z, world_y / world_z


if __name__ == "__main__":
  args = parser.parse_args()
  data_splits = ["train", "val", "test"]
  delim = "\t"

  h_files = glob.glob(os.path.join(args.h_path, "*.txt"))
  h_dict = {}  # image to world coordinates
  for h_file in h_files:
    scene = os.path.splitext(os.path.basename(h_file))[0]
    h_matrix = []
    with open(h_file, "r") as f:
      for line in f:
        h_matrix.append(line.strip().split(","))
    h_matrix = np.array(h_matrix, dtype="float")

    h_dict[scene] = h_matrix

  for data_split in data_splits:
    target_path = os.path.join(args.new_traj_path, data_split)
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    traj_files = glob.glob(os.path.join(args.traj_path, data_split, "*.txt"))

    for traj_file in tqdm.tqdm(traj_files):
      videoname = os.path.splitext(os.path.basename(traj_file))[0]
      scene = get_scene(videoname)

      target_file = os.path.join(target_path, "%s.txt" % videoname)

      H = h_dict[scene]

      new_data = []

      with open(traj_file, "r") as f:
        for line in f:
          fidx, pid, x, y = line.strip().split(delim)
          x, y = float(x), float(y)
          if scene == "0002":
            # all trajectory is under 1920x1080, but original 0002 is 1280x720
            x = x * (1280 / 1920.0)
            y = y * (720 / 1080.0)
          w_x, w_y = get_world_coordinates((x, y), H)
          new_data.append([float(fidx), float(pid), w_x, w_y])

      if args.translate_positive:
        min_x = np.amin(np.array(new_data)[:, 2])

        min_y = np.amin(np.array(new_data)[:, 3])

        till_pos_x = 0
        if min_x < 0:
          till_pos_x = abs(min_x)
        till_pos_y = 0
        if min_y < 0:
          till_pos_y = abs(min_y)
        new_data = [(fidx, pid, w_x + till_pos_x, w_y + till_pos_y)
                    for fidx, pid, w_x, w_y in new_data]


      with open(target_file, "w") as f:
        for fidx, pid, w_x, w_y in new_data:
          f.writelines(
              "%s%s%s%s%s%s%s\n" % (fidx, delim, pid, delim, w_x, delim, w_y))

