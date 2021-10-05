# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Get person box in HRnet json format."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=broad-except
import argparse
import cPickle as pickle
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("box_pickle")
parser.add_argument("hrnet_outpath")
parser.add_argument("kp_pickle")

if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.box_pickle, "r") as f:
    box_dict = pickle.load(f)

  videoname = os.path.splitext(os.path.basename(args.box_pickle))[0]

  kp_dict = {}
  count_failed = 0
  for framenum_pid in box_dict:
    framenum, pid = framenum_pid.split("_")
    framenum = int(framenum)
    pid = int(pid)

    # some kp may fail
    hrnet_outfile = os.path.join(args.hrnet_outpath,
                                 "%s_F_%08d_%d.npy" % (videoname,
                                                       framenum, pid))
    try:
      kp_npy = np.load(hrnet_outfile)
    except Exception as e:
      # print("Warning, %s failed" % hrnet_outfile)
      count_failed += 1
      continue

    kp_dict[framenum_pid] = kp_npy  # (17, 3)

  print("%s, total person instance %s, got keypoint %s, failed %s,"
        " fail rate %.3f" % (videoname, len(box_dict), len(kp_dict),
                             count_failed, count_failed/float(len(box_dict))))
  with open(args.kp_pickle, "w") as f:
    pickle.dump(kp_dict, f)
