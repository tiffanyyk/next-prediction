# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given annotation path and filelst, get object/traj/activity annotations."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import argparse
import os
import operator
import cPickle as pickle
import yaml
from tqdm import tqdm
from utils import activity2id
from utils import object2id
from utils import valid_box
from utils import modify_box
from utils import get_traj_point
from utils import actev_scene2imgsize
from utils import get_scene

parser = argparse.ArgumentParser()
parser.add_argument("filelst")
parser.add_argument("anno_path")
parser.add_argument("out_path")

# For running parallel jobs, set --job 4 --curJob k, where k=1/2/3/4
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="this script run job Num")

scene2imgsize = actev_scene2imgsize


def load_yml_file_without_meta(yml_file):
  """Load the ActEV YAML annotation files."""
  with open(yml_file, "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    # get the meta index first
    mi = -1
    for i in xrange(len(data)):
      if "meta" not in data[i]:
        mi = i
        break
    assert mi >= 0

    return data[mi:]


def load_tracks(track_file):
  """Load track object type information."""
  trackid2object_ = {}
  data = load_yml_file_without_meta(track_file)
  for one in data:
    one = one["types"]  # added in v1_update
    # v4, changed to - { types: { id1: 0 , cset3: { Person: 1.0 } } }
    if "obj_type" not in one:
      one["obj_type"] = one["cset3"].keys()[0]
      assert len(one["cset3"].keys()) == 1
    trackid2object_[int(one["id1"])] = one["obj_type"]
  return trackid2object_


def load_activities(act_file_, trackid2object_, activity2id_):
  """Load activities for each tracks."""
  personid2acts_ = {}
  data = load_yml_file_without_meta(act_file_)
  for one in data:
    one = one["act"]
    act_name = one["act2"]
    if isinstance(act_name, type({})):
      assert len(act_name.keys()) == 1
      act_name = act_name.keys()[0]
    # ignore other activities
    if act_name in activity2id_:
      act_classid = activity2id[act_name]
      for actor in one["actors"]:
        trackid_ = int(actor["id1"])
        if trackid_ in trackid2object_ and \
           (trackid2object_[trackid_] == "Person"):
          timespan = actor["timespan"][0]["tsr0"]
          start, end = timespan
          start, end = int(start), int(end)
          if trackid_ not in personid2acts_:
            personid2acts_[trackid_] = []
          personid2acts_[trackid_].append((start, end, act_classid))

  for trackid_ in personid2acts_:
    personid2acts_[trackid_].sort(key=operator.itemgetter(0))
  return personid2acts_


def load_boxes(box_file_, imgsize_):
  """Load bounding boxes."""
  boxes_ = []
  data = load_yml_file_without_meta(box_file_)
  for one in data:
    one = one["geom"]  # added in v1_update
    trackid_ = int(one["id1"])
    frame_index_ = int(one["ts0"])

    bbox_ = [float(a) for a in one["g0"].split()]

    src = one["src"]
    assert src == "truth", (src, one)

    # check box valid
    is_valid = valid_box(bbox_, imgsize_)
    if not is_valid:
      # modify box to be valid?
      bbox_ = modify_box(bbox_, imgsize_)
      assert valid_box(bbox_, imgsize_)
    # so box is [x1, y1, x2, y2]
    boxes_.append((trackid_, frame_index_, bbox_))

  return boxes_


if __name__ == "__main__":
  args = parser.parse_args()

  video_filenames = [line.strip() for line in open(args.filelst).readlines()]

  if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

  count = 0
  for video_filename in tqdm(video_filenames):
    count += 1
    if (count % args.job) != (args.curJob - 1):
      continue

    scene = get_scene(video_filename)
    imgsize = scene2imgsize[scene]

    box_file = os.path.join(args.anno_path, video_filename + ".geom.yml")
    type_file = os.path.join(args.anno_path, video_filename + ".types.yml")
    act_file = os.path.join(args.anno_path, video_filename + ".activities.yml")

    # load each track id and its trajectories
    origin_trackid2object = load_tracks(type_file)

    trackid2object = {trackid: origin_trackid2object[trackid]
                      for trackid in origin_trackid2object
                      if origin_trackid2object[trackid] in object2id}

    # load traj boxes for the trackid
    person_tracks = {}  # trackid -> boxes
    frameidx2boxes = {}  # each frame all boxes
    boxes = load_boxes(box_file, imgsize)

    for box in boxes:
      trackid, frame_index, bbox = box
      if trackid in trackid2object:
        if frame_index not in frameidx2boxes:
          frameidx2boxes[frame_index] = {
              "boxes": [],
              "classids": [],
              "trackids": []
          }
        frameidx2boxes[frame_index]["boxes"].append(bbox)
        frameidx2boxes[frame_index]["classids"].append(
            object2id[trackid2object[trackid]])
        frameidx2boxes[frame_index]["trackids"].append(trackid)
        # save the person track exclusively
        if trackid2object[trackid] == "Person":
          if trackid not in person_tracks:
            person_tracks[trackid] = []
          person_tracks[trackid].append({
              "f": frame_index,
              "box": bbox,
              "point": get_traj_point(bbox),
          })
    for personid in person_tracks:
      person_tracks[personid].sort(key=operator.itemgetter("f"))

    # load activities for the tracks we care
    # personid -> (start, end, act_classid)
    personid2acts = load_activities(act_file, trackid2object, activity2id)

    anno = {
        "person_tracks": person_tracks,
        "frameidx2boxes": frameidx2boxes,
        "personid2acts": personid2acts
    }
    target_file = os.path.join(args.out_path, "%s.npz" % (video_filename))
    with open(target_file, "w") as fw:
      pickle.dump(anno, fw)
