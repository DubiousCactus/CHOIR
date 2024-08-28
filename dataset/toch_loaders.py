import glob
import os
import pickle

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from dataset.contactpose import ContactPoseDataset
from utils.toch_utils import random_rotate_np

import math

class GRAB_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_folder, split, window_size=30, step_size=15, num_points=8000
    ):
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split

        files_clean = glob.glob(os.path.join(data_folder, split, "*.npy"))
        for f in files_clean:
            clip_clean = np.load(f)
            clip_pert = np.load(
                os.path.join(data_folder, split + "_pert", os.path.basename(f))
            )
            clip_len = (len(clip_clean) - window_size) // step_size + 1
            self.clips.append(
                (
                    self.len,
                    self.len + clip_len,
                    clip_pert,
                    [clip_clean["f9"], clip_clean["f11"], clip_clean["f10"]],
                )
            )
            self.len += clip_len
        self.clips.sort(key=lambda x: x[0])

    def __getitem__(self, index):
        for c in self.clips:
            if index < c[1]:
                break
        start_idx = (index - c[0]) * self.step_size
        data = c[2][start_idx : start_idx + self.window_size]
        corr_mask_gt, corr_pts_gt, corr_dist_gt = c[3][0], c[3][1], c[3][2]
        corr_mask_gt = corr_mask_gt[start_idx : start_idx + self.window_size]
        corr_pts_gt = corr_pts_gt[start_idx : start_idx + self.window_size]
        corr_dist_gt = corr_dist_gt[start_idx : start_idx + self.window_size]

        samp_ind = np.random.choice(list(range(self.num_points)), 4000, replace=False)

        object_pc = data["f3"].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data["f4"].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc, R = random_rotate_np(object_pc)

        object_pc = object_pc / np.max(np.sqrt(np.sum(object_pc[0] ** 2, axis=1)))

        object_normal = np.matmul(object_normal, R)
        object_corr_mask = (
            data["f9"].reshape(self.window_size, -1, 1).astype(np.float32)
        )
        object_corr_pts = (
            data["f11"].reshape(self.window_size, -1, 3).astype(np.float32)
        )
        object_corr_dist = (
            data["f10"].reshape(self.window_size, -1, 1).astype(np.float32)
        )

        corr_mask_gt = corr_mask_gt.reshape(self.window_size, -1, 1).astype(np.float32)
        corr_pts_gt = corr_pts_gt.reshape(self.window_size, -1, 3).astype(np.float32)
        corr_dist_gt = corr_dist_gt.reshape(self.window_size, -1, 1).astype(np.float32)

        # distance thresholding
        corr_mask_gt[corr_dist_gt > 0.1] = 0
        object_corr_mask[object_corr_dist > 0.1] = 0

        window_feat = np.concatenate(
            [
                object_pc,
                object_corr_mask,
                object_corr_pts,
                object_corr_dist,
                object_normal,
            ],
            axis=2,
        )[:, samp_ind[:2000]]

        object_pc_dec = object_pc[:, samp_ind[2000:]]
        object_normal_dec = object_normal[:, samp_ind[2000:]]
        corr_mask_gt = corr_mask_gt[:, samp_ind[2000:]]
        corr_pts_gt = corr_pts_gt[:, samp_ind[2000:]]
        corr_dist_gt = corr_dist_gt[:, samp_ind[2000:]]

        dec_cond = np.concatenate([object_pc_dec, object_normal_dec], axis=2)

        if np.random.uniform() > 0.5:
            window_feat = np.flip(window_feat, axis=0).copy()
            corr_mask_gt = np.flip(corr_mask_gt, axis=0).copy()
            corr_pts_gt = np.flip(corr_pts_gt, axis=0).copy()
            corr_dist_gt = np.flip(corr_dist_gt, axis=0).copy()
            dec_cond = np.flip(dec_cond, axis=0).copy()

        return np.concatenate(
            [window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, dec_cond], axis=2
        )

    def __len__(self):
        return self.len


class GRAB_Single_Frame(torch.utils.data.Dataset):
    def __init__(self, seq_path):
        self.data = np.load(seq_path)

    def __getitem__(self, index):
        num_points = self.data[index]["f3"].shape[0] // 3
        samp_ind = np.random.choice(list(range(num_points)), 2000)

        rhand_pc = self.data[index]["f0"].reshape(778, 3)
        object_pc = self.data[index]["f3"].reshape(-1, 3)[samp_ind]
        object_vn = self.data[index]["f4"].reshape(-1, 3)[samp_ind]
        object_corr_mask = self.data[index]["f9"].reshape(-1)[samp_ind]
        object_corr_pts = self.data[index]["f11"].reshape(-1, 3)[samp_ind]
        object_corr_dist = self.data[index]["f10"].reshape(-1)[samp_ind]

        object_global_orient = self.data[index]["f5"]
        object_transl = self.data[index]["f6"]
        object_id = self.data[index]["f7"]

        is_left = self.data[index]["f12"]

        return (
            rhand_pc,
            object_pc,
            object_corr_mask,
            object_corr_pts,
            object_corr_dist,
            object_global_orient,
            object_transl,
            object_id,
            object_vn,
            is_left,
        )

    def __len__(self):
        return len(self.data)


class ContactPoseDataset_Training(torch.utils.data.Dataset):
    def __init__(self, processed_root, split, num_points=8000):
        self.num_points = num_points
        self.window_size = 1
        self.data = []
        files_clean = glob.glob(os.path.join(processed_root, split, "corr_*.npy"))
        for f in tqdm(files_clean):
            grasp_clean = np.load(f)
            grasp_pert = np.load(
                os.path.join(processed_root, f"{split}_pert", os.path.basename(f))
            )
            # Should be: obj_corr_mask, obj_corr_pts, obj_corr_dist
            self.data.append(
                (
                    grasp_pert,
                    [grasp_clean["f10"], grasp_clean["f12"], grasp_clean["f11"]],
                )
            )

    def __getitem__(self, index):
        c = self.data[index]
        data = c[0][0]
        corr_mask_gt, corr_pts_gt, corr_dist_gt = c[1][0], c[1][1], c[1][2]
        corr_mask_gt, corr_pts_gt, corr_dist_gt = (
            corr_mask_gt[0],
            corr_pts_gt[0],
            corr_dist_gt[0],
        )
        # corr_mask_gt = corr_mask_gt[start_idx:start_idx+self.window_size]
        # corr_pts_gt = corr_pts_gt[start_idx:start_idx+self.window_size]
        # corr_dist_gt = corr_dist_gt[start_idx:start_idx+self.window_size]

        samp_ind = np.random.choice(list(range(self.num_points)), 4000, replace=False)

        object_pc = data["f4"].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data["f5"].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc, R = random_rotate_np(object_pc)

        object_pc = object_pc / np.max(np.sqrt(np.sum(object_pc[0] ** 2, axis=1)))

        object_normal = np.matmul(object_normal, R)
        object_corr_mask = (
            data["f10"].reshape(self.window_size, -1, 1).astype(np.float32)
        )
        object_corr_pts = (
            data["f12"].reshape(self.window_size, -1, 3).astype(np.float32)
        )
        object_corr_dist = (
            data["f11"].reshape(self.window_size, -1, 1).astype(np.float32)
        )

        corr_mask_gt = corr_mask_gt.reshape(self.window_size, -1, 1).astype(np.float32)
        corr_pts_gt = corr_pts_gt.reshape(self.window_size, -1, 3).astype(np.float32)
        corr_dist_gt = corr_dist_gt.reshape(self.window_size, -1, 1).astype(np.float32)

        # distance thresholding
        corr_mask_gt[corr_dist_gt > 0.1] = 0
        object_corr_mask[object_corr_dist > 0.1] = 0

        window_feat = np.concatenate(
            [
                object_pc,
                object_corr_mask,
                object_corr_pts,
                object_corr_dist,
                object_normal,
            ],
            axis=2,
        )[:, samp_ind[:2000]]

        object_pc_dec = object_pc[:, samp_ind[2000:]]
        object_normal_dec = object_normal[:, samp_ind[2000:]]
        corr_mask_gt = corr_mask_gt[:, samp_ind[2000:]]
        corr_pts_gt = corr_pts_gt[:, samp_ind[2000:]]
        corr_dist_gt = corr_dist_gt[:, samp_ind[2000:]]

        dec_cond = np.concatenate([object_pc_dec, object_normal_dec], axis=2)

        if np.random.uniform() > 0.5:
            window_feat = np.flip(window_feat, axis=0).copy()
            corr_mask_gt = np.flip(corr_mask_gt, axis=0).copy()
            corr_pts_gt = np.flip(corr_pts_gt, axis=0).copy()
            corr_dist_gt = np.flip(corr_dist_gt, axis=0).copy()
            dec_cond = np.flip(dec_cond, axis=0).copy()

        return np.concatenate(
            [window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, dec_cond], axis=2
        )

    def __len__(self):
        return len(self.data)


class ContactPose_Single_Frame(torch.utils.data.Dataset):
    def __init__(self, grasp_path):
        assert os.path.isfile(grasp_path)
        self.data = np.load(grasp_path)  # Load a single grasp file

    def __getitem__(self, index):
        num_points = self.data[index]["f4"].shape[0] // 3
        samp_ind = np.random.choice(list(range(num_points)), 2000)

        rhand_pc = self.data[index]["f0"].reshape(778, 3)
        object_pc = self.data[index]["f4"].reshape(-1, 3)[samp_ind]
        object_vn = self.data[index]["f5"].reshape(-1, 3)[samp_ind]
        object_corr_mask = self.data[index]["f10"].reshape(-1)[samp_ind]
        object_corr_pts = self.data[index]["f12"].reshape(-1, 3)[samp_ind]
        object_corr_dist = self.data[index]["f11"].reshape(-1)[samp_ind]

        object_global_orient = self.data[index]["f6"]
        object_transl = self.data[index]["f7"]
        object_id = self.data[index]["f8"]

        is_left = self.data[index]["f13"]

        return (
            rhand_pc,
            object_pc,
            object_corr_mask,
            object_corr_pts,
            object_corr_dist,
            object_global_orient,
            object_transl,
            object_id,
            object_vn,
            is_left,
        )

    def __len__(self):
        return len(self.data)


class ContactPoseDataset_Eval(torch.utils.data.Dataset):
    def __init__(self, processed_root, **kwargs):
        self.n_noisy_variations = 4
        self.samples, self.gt, self.object_contact_maps = [], [], []
        files_clean = sorted(glob.glob(os.path.join(processed_root, "test", "corr_*.npy")))
        for f in tqdm(files_clean):
            grasp_clean = np.load(f)
            grasp_pert = np.load(
                os.path.join(processed_root, f"test_pert", os.path.basename(f))
            )
            # Should be: obj_corr_mask, obj_corr_pts, obj_corr_dist
            self.samples.append(
                (
                    grasp_pert,
                    [grasp_clean["f10"], grasp_clean["f12"], grasp_clean["f11"]],
                )
            )
            self.gt.append(
                {
                    "theta": grasp_clean["f3"].copy(),
                    "beta": np.zeros((1, 10)),
                    "rot": grasp_clean["f1"].copy(),
                    "trans": grasp_clean["f2"].copy(),
                    "verts": grasp_clean["f0"].reshape(778, 3).copy(),
                }
            )
        # Now let's try to load the contact maps under the assumption that they are in the exact
        # same order as the samples/gt:
        path_to_test_data = os.path.join(
            processed_root,
            "dataset_51-participants_from-0.8_to-1.0_split_8000-obj-pts_right-hand.pkl",
        )
        with open(path_to_test_data, "rb") as f:
            compressed_pkl = f.read()
            objects_w_contacts, _, _, _ = pickle.loads(compressed_pkl)
            self.object_contact_maps.extend(objects_w_contacts)

        assert len(self.samples) > 0, "No data found!"
        assert len(self.samples) == len(self.gt), "Mismatch in data and ground truth!"
        assert len(self.object_contact_maps) * self.n_noisy_variations == len(
            self.samples
        ), "Mismatch in data and object paths!"

    def register_sampling_inst(self, *args, **kwargs):
        pass

    @property
    def theta_dim(self):
        return 18

    @property
    def bps_dim(self):
        return 0

    @property
    def name(self):
        return "toch_contactpose"

    @property
    def bps(self):
        return None

    @property
    def anchor_indices(self):
        return None

    @property
    def remap_bps_distances(self):
        return False

    @property
    def exponential_map_w(self):
        return None

    @property
    def base_unit(self):
        return ContactPoseDataset.base_unit

    def set_eval_mode(self, eval_mode):
        pass

    def set_observations_number(self, num_observations):
        pass

    @property
    def min_pts(self):
        return 0

    @property
    def max_ctx_pts(self):
        return 0

    @property
    def is_right_hand_only(self):
        return True

    @property
    def center_on_object_com(self):
        return True

    @property
    def eval_observation_plateau(self):
        return False

    def __getitem__(self, index):
        noisy_sample_feat, gt_sample_feat = self.samples[index]
        label = self.gt[index]
        mesh_pth = self.object_contact_maps[math.floor(index//self.n_noisy_variations)]
        num_points = noisy_sample_feat[0]["f4"].shape[0] // 3
        samp_ind = np.random.choice(list(range(num_points)), 2000)

        rhand_pc = noisy_sample_feat[0]["f0"].reshape(778, 3)
        object_pc = noisy_sample_feat[0]["f4"].reshape(-1, 3)[samp_ind]
        object_vn = noisy_sample_feat[0]["f5"].reshape(-1, 3)[samp_ind]
        object_corr_mask = noisy_sample_feat[0]["f10"].reshape(-1)[samp_ind]
        object_corr_pts = noisy_sample_feat[0]["f12"].reshape(-1, 3)[samp_ind]
        object_corr_dist = noisy_sample_feat[0]["f11"].reshape(-1)[samp_ind]

        object_global_orient = noisy_sample_feat[0]["f6"]
        object_transl = noisy_sample_feat[0]["f7"]
        object_id = noisy_sample_feat[0]["f8"]

        is_left = noisy_sample_feat[0]["f13"]

        sample = {
            "toch_features": (
                rhand_pc,
                object_pc,
                object_corr_mask,
                object_corr_pts,
                object_corr_dist,
                object_global_orient,
                object_transl,
                object_id,
                object_vn,
                is_left,
            ),
            "scalar": 1.0,
        }
        return sample, label, mesh_pth

    def __len__(self):
        return len(self.samples)
