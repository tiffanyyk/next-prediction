# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Parameter free constant velocity experiment."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import argparse
import math
import os
import sys
import tqdm
import numpy as np

# The next-prediction code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pred_utils

parser = argparse.ArgumentParser()

parser.add_argument("prepropath")

parser.add_argument("exp_type", type=int, default=1,
                    help="1: last two timestep;2: mean velocity")

parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--is_actev", action="store_true",
                    help="is actev/virat dataset, has activity info")

if __name__ == "__main__":
  args = parser.parse_args()

  # some hyper-parameters to keep it consistent with previous exps
  args.batch_size = 64
  args.add_activity = False

  # has to be 2,4 to match the scene CNN strides
  args.scene_grid_strides = (2, 4)
  args.scene_grids = []
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  if args.is_actev:
    args.virat_mov_actids = [
        pred_utils.activity2id["activity_walking"],
        pred_utils.activity2id["activity_running"],
        pred_utils.activity2id["Riding"],
    ]
    args.traj_cats = [
        ["static", 0],
        ["mov", 1],
    ]
    args.scenes = ["0000", "0002", "0400", "0401", "0500"]

  # --------------------------------------------------------------

  test_data = pred_utils.read_data(args, "test")

  l2dis = []  # [num_example, each_timestep]

  # show the evaluation per trajectory class if actev experiment
  if args.is_actev:
    l2dis_cats = [[] for i in xrange(len(args.traj_cats))]
    # added 06/2019,
    # show per-scene ADE/FDE for ActEV dataset
    # for leave-one-scene-out experiment
    l2dis_scenes = [[] for i in xrange(len(args.scenes))]

  num_batches_per_epoch = int(
      math.ceil(test_data.num_examples / float(args.batch_size)))

  for evalbatch in tqdm.tqdm(test_data.get_batches(args.batch_size, \
    full=True, shuffle=False), total=num_batches_per_epoch, ascii=True):

    _, batch = evalbatch

    assert len(batch.data["obs_traj_rel"]) == args.batch_size

    this_actual_batch_size = batch.data["original_batch_size"]

    # get the velocity from the observation data
    # [N, T_obs, 2]
    obs_traj_velocity = np.array(
        batch.data["obs_traj_rel"][:this_actual_batch_size])

    # constant_velocity [N, 2]
    if args.exp_type == 1:
      constant_velocity = obs_traj_velocity[:, -1, :]
    else:
      constant_velocity = np.mean(obs_traj_velocity, axis=1)

    # [N, T_pred, 2]
    pred_out = np.tile(
        np.expand_dims(constant_velocity, axis=1), [1, args.pred_len, 1])

    d = []

    for i, (obs_traj_gt, pred_traj_gt) in enumerate(
        zip(batch.data["obs_traj"], batch.data["pred_traj"])):
      if i >= this_actual_batch_size:
        break
      # the output is relative coordinates
      this_pred_out = pred_out[i][:, :2]  # [T2, 2]
      # [T2,2]
      this_pred_out_abs = pred_utils.relative_to_abs(this_pred_out,
                                                     obs_traj_gt[-1])
      # get the errors
      assert this_pred_out_abs.shape == this_pred_out.shape, (
          this_pred_out_abs.shape, this_pred_out.shape)

      # [T2, 2]
      diff = pred_traj_gt - this_pred_out_abs
      diff = diff**2
      diff = np.sqrt(np.sum(diff, axis=1))  # [T2]

      d.append(diff)

      if args.is_actev:
        traj_cat_id = batch.data["trajidx2catid"][i]
        l2dis_cats[traj_cat_id].append(diff)  # [T2]
        # per-scene eval
        traj_key = batch.data["traj_key"][i]  # videoname_frameidx_personid
        # videoname has '_'
        videoname = traj_key[::-1].split("_", 2)[-1][::-1]
        scene = pred_utils.get_scene(videoname)  # 0000/0002, etc.
        l2dis_scenes[args.scenes.index(scene)].append(diff)

    l2dis += d

  # average displacement
  ade = [t for o in l2dis for t in o]
  # final displacement
  fde = [o[-1] for o in l2dis]
  perf = {"ade": np.mean(ade),
          "fde": np.mean(fde),
          "grid1_acc": None,
          "grid2_acc": None,
          "act_ap": None}

  # show ade and fde for different traj category
  if args.is_actev:
    # per-traj-class eval
    for cat_id, (cat_name, _) in enumerate(args.traj_cats):
      diffs = l2dis_cats[cat_id]
      ade = [t for l in diffs for t in l]
      fde = [l[-1] for l in diffs]
      perf.update({
          ("%s_ade" % cat_name): np.mean(ade) if ade else 0.0,
          ("%s_fde" % cat_name): np.mean(fde) if fde else 0.0,
      })

    # per-scene eval
    for scene_id, scene in enumerate(args.scenes):
      diffs = l2dis_scenes[scene_id]
      ade = [t for l in diffs for t in l]
      fde = [l[-1] for l in diffs]
      perf.update({
          ("%s_ade" % scene): np.mean(ade) if ade else 0.0,
          ("%s_fde" % scene): np.mean(fde) if fde else 0.0,
      })

  print("performance:")
  numbers = []
  for k in sorted(perf):
    print("%s, %s" % (k, perf[k]))
    numbers.append("%s" % perf[k])
  print(" ".join(sorted(perf.keys())))
  print(" ".join(numbers))
