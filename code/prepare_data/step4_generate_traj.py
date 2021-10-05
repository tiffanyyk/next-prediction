# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Generate trajectory files and scene, person box, other box, activity files."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import argparse
import os
import operator
import numpy as np
import cPickle as pickle
from tqdm import tqdm
from glob import glob
from utils import activity2id
from utils import actev_scene2imgsize
from utils import get_scene


parser = argparse.ArgumentParser()
parser.add_argument("npzpath")
parser.add_argument("split_path")
parser.add_argument("out_path")
parser.add_argument("--drop_frame", default=1, type=int,
                    help="drop frame to match different fps, assuming "
                         "the virat fps is 30fps, so to get 2.5fps, "
                         "need to drop 12 frames every time")

parser.add_argument("--scene_feat_path",
                    help="the scene segmentation output path,"
                         "under it should be frame_name.npy")

# the following are the output paths
parser.add_argument("--scene_map_path",
                    help="frameidx mapping to actual scene feature file output")

parser.add_argument("--person_box_path",
                    help="Person box output")

parser.add_argument("--other_box_path",
                    help="Other object box output")

parser.add_argument("--activity_path",
                    help="activity annotation output")

# for ETH/UCY you need to write your own video size mapping
# In the PeekingFuture paper we resize ETH/UCY to 720x576 to extract features
scene2imgsize = actev_scene2imgsize

actid2name = {activity2id[n]: n for n in activity2id}


def resize_xy(xy, vname, resize_w, resize_h):
  """Resize the xy coordinates."""
  x_, y_ = xy
  w, h = scene2imgsize[get_scene(vname)]
  diff_w = resize_w / float(w)
  diff_h = resize_h / float(h)
  x_ *= diff_w
  y_ *= diff_h

  # normalize coordinates?
  return [x_, y_]


def resize_box(box, vname, resize_w, resize_h):
  """Resize the box coordintates."""
  x1, y1, x2, y2 = [float(o) for o in box]

  w, h = scene2imgsize[get_scene(vname)]
  diff_w = resize_w / float(w)
  diff_h = resize_h / float(h)

  x1 *= diff_w
  x2 *= diff_w
  y1 *= diff_h
  y2 *= diff_h
  return [x1, y1, x2, y2]


# frame_lst is [(videoname,frameidx)], assume sorted by the frameidx
def get_nearest(frame_lst_, frame_idx):
  """Since we don't run scene seg on every frame, we want to find the nearest one."""
  frame_idxs = np.array([i_ for _, i_ in frame_lst_])
  cloests_idx = (np.abs(frame_idxs - frame_idx)).argmin()
  vname, closest_frame_idx = frame_lst_[cloests_idx]
  return vname, closest_frame_idx, cloests_idx


def get_act_list(act_data, frameidx, bgid):
  """Given a frameidx, get this person' activities."""

  # act_data is a list of sorted (start,end,actclassid)
  # return current act list,
  current_act_list = [(actid, e - frameidx) for s, e, actid in act_data
                      if (frameidx >= s) and (frameidx <= e)]
  current_act_list.sort(key=operator.itemgetter(1))  # dist to current act's end
  current_actid_list_ = [actid for actid, _ in current_act_list]
  current_dist_list_ = [dist for _, dist in current_act_list]

  if not current_act_list:
    current_actid_list_, current_dist_list_ = [bgid], [-1]

  future_act_list = [(actid, s - frameidx) for s, e, actid in act_data
                     if frameidx < s]
  future_act_list.sort(key=operator.itemgetter(1))

  if not future_act_list:
    return (current_actid_list_, current_dist_list_, [bgid], [-1])

  # only the nearest future activity?
  # smallest_dist = future_act_list[0][1]
  # future_act_list = [(actid,dist) for actid, dist in future_act_list
  #                     if dist == smallest_dist]

  future_actid_list_ = [actid for actid, _ in future_act_list]
  future_dist_list_ = [dist for _, dist in future_act_list]

  return (current_actid_list_, current_dist_list_,
          future_actid_list_, future_dist_list_)


def check_traj(newdata_, vname):
  """Check and filter data."""
  checkdata = np.array(newdata_, dtype="float")
  frames_ = np.unique(checkdata[:, 0]).tolist()
  checked_data_ = []
  for frame_ in frames_:
    # all personid in this frame
    this_frame_data = checkdata[frame_ == checkdata[:, 0], :]  # [K,4]
    ped_ids = this_frame_data[:, 1]
    unique_ped_ids, unique_idxs = np.unique(ped_ids, return_index=True)
    if len(ped_ids) != len(unique_ped_ids):
      tqdm.write("\twarning, %s frame %s has duplicate person annotation person"
                 " ids: %s/%s, removed the duplicate ones"
                 % (vname, frame_, len(unique_ped_ids), len(ped_ids)))

      this_frame_data = this_frame_data[unique_idxs]

    for f_, p_, x_, y_ in this_frame_data:
      checked_data_.append((f_, p_, x_, y_))
  checked_data_.sort(key=operator.itemgetter(0))
  return checked_data_


if __name__ == "__main__":
  args = parser.parse_args()

  # Hard coded for ActEV experiment.
  # :P
  args.resize = True
  args.resize_h = 1080
  args.resize_w = 1920

  filelst = {
      "train": [os.path.splitext(os.path.basename(line.strip()))[0]
                for line in open(os.path.join(args.split_path,
                                              "train.lst"), "r").readlines()],
      "val": [os.path.splitext(os.path.basename(line.strip()))[0]
              for line in open(os.path.join(args.split_path,
                                            "val.lst"), "r").readlines()],
      "test": [os.path.splitext(os.path.basename(line.strip()))[0]
               for line in open(os.path.join(args.split_path,
                                             "test.lst"), "r").readlines()],
  }

  for split in tqdm(filelst, ascii=True):
    out_path = os.path.join(args.out_path, split)

    if not os.path.exists(out_path):
      os.makedirs(out_path)

    if not os.path.exists(os.path.join(args.person_box_path, split)):
      os.makedirs(os.path.join(args.person_box_path, split))

    if not os.path.exists(os.path.join(args.other_box_path, split)):
      os.makedirs(os.path.join(args.other_box_path, split))

    if not os.path.exists(os.path.join(args.activity_path, split)):
      os.makedirs(os.path.join(args.activity_path, split))

    scene_map_path = os.path.join(args.scene_map_path, split)
    if not os.path.exists(scene_map_path):
      os.makedirs(scene_map_path)

    for videoname in tqdm(filelst[split]):
      npzfile = os.path.join(args.npzpath, "%s.npz" % videoname)

      data = np.load(npzfile, allow_pickle=True)

      # each frame's all boxes, for getting other boxes
      frameidx2boxes = data["frameidx2boxes"]

      # personId -> all related activity with timespan, sorted by timespan start
      # (start, end, act_classid)
      personid2acts = data["personid2acts"]

      # load all the frames for this video first
      frame_lst = glob(os.path.join(args.scene_feat_path,
                                    "%s_F_*.npy"%videoname))
      assert frame_lst
      frame_lst = [(os.path.basename(frame),
                    int(os.path.basename(frame).split(".")[0].split("_F_")[-1]))
                   for frame in frame_lst]

      frame_lst.sort(key=operator.itemgetter(1))

      newdata = []  # (frame_id, person_id,x,y) # all float
      # key is frameidx, person_id
      scene_data = {}  # frame ->
      person_box_data = {}  # key -> person boxes
      other_box_data = {}  # key -> other box + boxclassids
      activity_data = {}  # key ->
      for person_id in data["person_tracks"]:
        for d in data["person_tracks"][person_id]:
          # resize or normalize
          d["point"] = resize_xy(d["point"], videoname,
                                 args.resize_w, args.resize_h)
          newdata.append((
              d["f"],
              float(person_id),
              d["point"][0],
              d["point"][1]
          ))

          person_key = "%d_%d" % (d["f"], person_id)

          # 1. get person boxes
          person_box = resize_box(d["box"], videoname,
                                  args.resize_w, args.resize_h)

          person_box_data[person_key] = person_box
          # done 1

          # 2. get other boxes
          all_boxes = frameidx2boxes[d["f"]]

          # remove itself in the object boxes
          this_person_idx = all_boxes["trackids"].index(person_id)

          all_other_boxes = [all_boxes["boxes"][i]
                             for i in xrange(len(all_boxes["boxes"]))
                             if i != this_person_idx]
          all_other_boxclassids = [all_boxes["classids"][i]
                                   for i in xrange(len(all_boxes["classids"]))
                                   if i != this_person_idx]

          # resize the box if needed
          for i in xrange(len(all_other_boxes)):
            all_other_boxes[i] = resize_box(all_other_boxes[i], videoname,
                                            args.resize_w, args.resize_h)

          other_box_data[person_key] = (all_other_boxes, all_other_boxclassids)
          # done 2

          # 3. get activity annotations
          if person_id in personid2acts:
            this_person_acts = personid2acts[person_id]

            # get the current activitylist, future activity list
            # and timestep to future activities
            current_actid_list, current_dist_list, future_actid_list, \
            future_dist_list = get_act_list(this_person_acts,
                                            d["f"], activity2id["BG"])
          else:
            # so some person has no activity?
            current_actid_list, current_dist_list, future_actid_list, \
            future_dist_list = ([activity2id["BG"]], [-1], [activity2id["BG"]],
                                [-1])

          activity_data[person_key] = (current_actid_list,
                                       current_dist_list,
                                       future_actid_list,
                                       future_dist_list)
          for lst in activity_data[person_key]:
            assert lst
          # done 3

          # 4. get scene segmentation features
          # find the nearest scene out file to this frame
          this_frame_idx = d["f"]
          target_file, target_frame_idx, idx = get_nearest(frame_lst,
                                                           this_frame_idx)
          # print("this frame %s, found target_file %s and target frame idx %s,"
          #      "since the frame list is like %s"
          #      % (this_frame_idx, target_file, target_frame_idx,
          #         frame_lst[idx-2:idx+2]))

          scene_data[d["f"]] = target_file

          # done 4

      newdata.sort(key=operator.itemgetter(0))

      if args.drop_frame > 1:
        frames = np.unique([one[0] for one in newdata]).tolist()
        # uniformly drop frames
        frames_to_keep = frames[::args.drop_frame]
        drop_newdata = [one for one in newdata if one[0] in frames_to_keep]

        newdata = drop_newdata

      # do a data check, each frame"s person id should be unique
      # check and filter
      checked_data = check_traj(newdata, videoname)
      if len(checked_data) != len(newdata):
        tqdm.write("checked data vs original data:%s/%s" % (len(checked_data),
                                                            len(newdata)))
      newdata = checked_data

      desfile = os.path.join(args.out_path, split, "%s.txt" % videoname)

      delim = "\t"
      with open(desfile, "w") as f:
        for i, p, x, y in newdata:
          f.writelines("%d%s%.1f%s%.6f%s%.6f\n" % (i, delim, p, delim, x,
                                                   delim, y))

      with open(os.path.join(args.person_box_path,
                             split, "%s.p" % videoname), "w") as f:
        pickle.dump(person_box_data, f)

      with open(os.path.join(args.other_box_path,
                             split, "%s.p" % videoname), "w") as f:
        pickle.dump(other_box_data, f)

      with open(os.path.join(args.activity_path,
                             split, "%s.p" % videoname), "w") as f:
        pickle.dump(activity_data, f)

      with open(os.path.join(args.scene_map_path,
                             split, "%s.p" % videoname), "w") as f:
        pickle.dump(scene_data, f)
