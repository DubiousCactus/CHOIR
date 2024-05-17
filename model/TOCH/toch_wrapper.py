#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
TOCH wrapper around the model but w/ TTO as well, only for evaluation.
"""


import copy
import os
import os.path as osp
import pickle as pkl
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import conf.project as project_conf
from model.TOCH import TemporalPointAE
from utils import colorize

try:
    from psbody.mesh import Mesh
except ImportError:
    raise ImportError(
        "Please install the psbody-mesh package to evaluate TOCH. Otherwise, don't import this file."
    )


def seal(mesh_to_seal):
    circle_v_id = np.array(
        [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
        dtype=np.int32,
    )
    center = (mesh_to_seal.v[circle_v_id, :]).mean(0)

    sealed_mesh = copy.copy(mesh_to_seal)
    sealed_mesh.v = np.vstack([mesh_to_seal.v, center])
    center_v_id = sealed_mesh.v.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i - 1], circle_v_id[i], center_v_id]
        sealed_mesh.f = np.vstack([sealed_mesh.f, new_faces])
    return sealed_mesh


class TOCHInference(torch.nn.Module):
    def __init__(
        self,
        processed_data_path: str,
        mano_models_path: str,
        model_path: Optional[str] = None,
        latent_size: int = 64,
        num_init: int = 1,
    ):
        super().__init__()
        self.coarse_lr = 0.1
        self.fine_lr = 0.1
        self.num_fine_iter = 2000
        self.num_coarse_iter = 100
        self.single_modality = "noisy_pair"
        self.num_init = num_init
        print(
            colorize(
                "Creating TOCH inference wrapper for ContactPose only!",
                project_conf.ANSI_COLORS["red"],
            )
        )
        print(
            colorize(f"Manually loading TOCH model from {model_path}...", project_conf.ANSI_COLORS["cyan"])
        )
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        self.model = TemporalPointAE(
            input_dim=11, latent_dim=latent_size, window_size=1
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.mano_mesh = None
        # load MANO
        with open(os.path.join(mano_models_path, "MANO_RIGHT.pkl"), "rb") as f:
            self.mano = pkl.load(f, encoding="latin-1")
            mano_mesh = Mesh(self.mano["v_template"], self.mano["f"])
            J_regressor = torch.tensor(self.mano["J_regressor"].todense()).float()
        # Loaded from the data/grab/scale_center.pkl file of the repo:
        #scale, center = 5.180721556271635, np.array(
        #    [-0.09076019, -0.02022504, -0.05842724]
        #)
        #mano_mesh.v = mano_mesh.v * scale + center
        with open("/home/moralest/toch/data/grab/scale_center.pkl", "rb") as f:
            scale, center = pkl.load(f)
            mano_mesh.v = mano_mesh.v * scale + center
            self.mano_mesh = seal(mano_mesh)

        object_paths = []
        for f in os.listdir(processed_data_path):
            if f.endswith(".pkl") and f.startswith("dataset"):
                dataset_path = osp.join(processed_data_path, f)
                with open(dataset_path, "rb") as f:
                    compressed_pkl = f.read()
                    objects_w_contacts, _, _, _ = pkl.loads(compressed_pkl)
                    object_paths.extend(objects_w_contacts)
        object_paths = list(set(object_paths))

        self.id2objmesh = {}
        with open(osp.join(processed_data_path, "objname2id.pkl"), "rb") as f:
            objname2id = pkl.load(f)
            for objname, objid in objname2id.items():
                obj_path = None
                for p in object_paths:
                    if objname in p:
                        obj_path = p  # We just need any match, doesn't matter
                        break
                else:
                    # Actually it's possible that this split doesn't have this object. We just skip it.
                    continue
                    # raise FileNotFoundError("Object {} not found in dataset".format(objname))
                self.id2objmesh[objid] = obj_path

        # setup MANO layer
        # I won't change that to my layer because their hyperparameters were turned on axis-ang with 24
        # components, which would give different convergence speed and results on my layer. The means
        # don't matter here anyway.
        self.mano_models_path = mano_models_path
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side="right",
            mano_root=mano_models_path,
            ncomps=24,
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        )

    def forward(
        self,
        input_rhand_pc,
        obj_pc,
        obj_corr_mask,
        obj_corr_pts,
        obj_corr_dist,
        obj_rot,
        obj_transl,
        obj_id,
        obj_vn,
        is_left,
        gt=None
    ):
        torch.set_grad_enabled(True)
        # What to return (bare minimum):
        # { "verts": , "joints": , "anchors": }
        # I think the anchors can be ignored.
        assert (
            len(input_rhand_pc.shape) == 2 and len(obj_pc.shape) == 2
        ), "ContactPose is for single frame optimization only. Inputs shouldn't be batched."
        input_rhand_pcs = []
        object_verts = []
        input_features = []
        Rs = []
        ts = []

        obj_corr_mask[obj_corr_dist > 0.1] = 0
        obj_mesh = Mesh(filename=self.id2objmesh[obj_id.item()])
        obj_verts = np.dot(
            obj_mesh.v, R.from_rotvec(obj_rot).as_matrix()
        ) + obj_transl.reshape(1, 3)
        obj_verts -= obj_verts.mean(axis=0, keepdims=True)

        obj_pc_variance = np.max(np.sqrt(np.sum(obj_pc**2, axis=1)))
        obj_pc = obj_pc / obj_pc_variance
        object_pc = torch.tensor(obj_pc, dtype=torch.float32)

        object_vn = torch.tensor(obj_vn, dtype=torch.float32)
        object_corr_mask = torch.tensor(obj_corr_mask, dtype=torch.float).unsqueeze(-1)
        object_corr_pts = torch.tensor(obj_corr_pts, dtype=torch.float)
        object_corr_dist = torch.tensor(obj_corr_dist, dtype=torch.float).unsqueeze(-1)

        if is_left:
            raise NotImplementedError("Left hand not supported")
            input_rhand_pc[..., 0] = -input_rhand_pc[..., 0]
        input_rhand_pcs.append(input_rhand_pc)
        object_verts.append(obj_verts)

        Rs.append(R.from_rotvec(obj_rot).as_matrix())
        ts.append(obj_transl.reshape(1, 3))

        input_features.append(
            torch.cat(
                [
                    object_pc,
                    object_corr_mask,
                    object_corr_pts,
                    object_corr_dist,
                    object_vn,
                ],
                dim=1,
            )
        )
        assert (
            len(input_features) == 1
        ), "ContactPose is for single frame optimization only. This should never be more than 1."
        # =================== Run the model inference on the GPU ====================
        device = next(self.model.parameters()).device
        batched_input = torch.stack(input_features, dim=0).unsqueeze(
            0
        ).to(device)  # Batch of 1 element with 1 frame
        #print(f"batched_input: {batched_input.shape}")

        with torch.no_grad():
            corr_mask_output, corr_pts_output, corr_dist_output = self.model(
                batched_input
            )
        #print(f"corr_mask_output: {corr_mask_output.shape}, corr_pts_output: {corr_pts_output.shape}, corr_dist_output: {corr_dist_output.shape}")
        # == Move everything back to the CPU so I don't have to migrate the original code from numpy to torch ==
        device = torch.device("cpu")
        corr_mask_output, corr_pts_output, corr_dist_output = corr_mask_output.to(device), corr_pts_output.to(device), corr_dist_output.to(device)
        batched_input = batched_input.to(device)

        data_collection = []
        object_contact_pts = []
        dist_weights = []

        assert (
            corr_mask_output.size(0) == 1
            and corr_pts_output.size(0) == 1
            and corr_dist_output.size(0) == 1
        ), "ContactPose is for single frame optimization only. This should never be more than 1."

        corr_mask_pred = torch.sigmoid(corr_mask_output[0]).numpy() > 0.5
        corr_pts_pred = corr_pts_output[0].numpy()

        obj_corr_pts = corr_pts_pred[0]
        obj_corr_mask = corr_mask_pred[0]

        object_pc = batched_input[0][0, :, :3] * obj_pc_variance
        object_vn = batched_input[0][0, :, -3:]
        object_corr_dist = corr_dist_output[0][0] * 0.1
        object_corr_mask = torch.from_numpy(obj_corr_mask)

        mano_mesh = copy.deepcopy(self.mano_mesh)

        #print(f"mano_mesh verts:{ mano_mesh.v[..., :10, :]}")
        #print(f"template mano mesh verts: {mano_mesh.v.shape}")

        closest_face, closest_points = mano_mesh.closest_faces_and_points(
            obj_corr_pts[obj_corr_mask]
        )
        # What this does is compute_aabb_tree().nearest(vertices) where vertices is arg to closest_faces_and_points.
        vert_ids, bary_coords = mano_mesh.barycentric_coordinates_for_points(
            closest_points, closest_face.astype("int32")
        )
        vert_ids = torch.from_numpy(vert_ids.astype(np.int64)).view(-1)
        bary_coords = torch.from_numpy(bary_coords).float()
        obj_contact_pts = (object_pc + object_corr_dist.unsqueeze(1) * object_vn)[
            object_corr_mask
        ]
        data_collection.append((vert_ids, bary_coords))
        object_contact_pts.append(obj_contact_pts)
        object_contact_pts = (
            torch.cat(object_contact_pts, dim=0)
            .unsqueeze(0)
            .repeat(self.num_init, 1, 1)
        )

        circle_v_id = torch.tensor(
            [
                108,
                79,
                78,
                121,
                214,
                215,
                279,
                239,
                234,
                92,
                38,
                122,
                118,
                117,
                119,
                120,
            ],
            dtype=torch.long,
        )

        num_frames = 1
        # initialize variables
        #print("====================== COARSE OPTIMIZATION ======================")
        device = next(self.model.parameters()).device # Back to GPU!
        device = torch.device("cpu")
        beta_var = torch.randn([self.num_init, 10]).to(device)
        # first 3 global orientation
        rot_var = torch.randn([self.num_init * num_frames, 3]).to(device)
        theta_var = torch.randn([self.num_init * num_frames, 24]).to(device)
        transl_var = torch.randn([self.num_init * num_frames, 3]).to(device)
        beta_var.requires_grad_(True)
        rot_var.requires_grad_(True)
        theta_var.requires_grad_(True)
        transl_var.requires_grad_(True)
        object_contact_pts = object_contact_pts.to(device)
        #print(f"theta_var: {theta_var.shape}, transl_var: {transl_var.shape}")
        self.mano_layer = self.mano_layer.to(device)

        #print(f"loading MANO from {self.mano_models_path}")
        mano_layer = ManoLayer(
            flat_hand_mean=True,
            side="right",
            mano_root=self.mano_models_path,
            ncomps=24,
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(device)
        #print(mano_layer)
        #print(f"transl_var: {transl_var[0]}")
        #print(f"object vertices: {object_verts[0].shape}")
        #print(f"object vertices: {object_verts[0][:10]}")


        # coarse optimization loop
        num_iters = self.num_coarse_iter
        opt = torch.optim.Adam([rot_var, transl_var], lr=self.coarse_lr)
        for i in range(num_iters):
            opt.zero_grad()
            hand_verts, _ = mano_layer(
                torch.cat([rot_var, theta_var], dim=-1),
                beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10),
                transl_var,
            )
            hand_verts = hand_verts.view(self.num_init, num_frames, 778, 3)# * 0.001
            
            #print(f"saving hand mesh: {hand_verts.shape}")
            #hand_mesh = Mesh(v=hand_verts[0, 0].detach().cpu().numpy(), f=self.mano["f"])
            #object_mesh = Mesh(v=object_verts[0], f=obj_mesh.f)
            #object_mesh.write_ply(os.path.join("visu", "object.ply"))
            #hand_mesh.write_ply(os.path.join("visu", "first_step_hand_stage1.ply"))
            #exit(1)

            center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
            hand_verts = torch.cat([hand_verts, center], dim=2)

            pred_contact_pts = []
            for j in range(self.num_init):
                for k in range(num_frames):
                    vert_ids = data_collection[k][0].to(device)
                    bary_coords = data_collection[k][1].to(device)
                    pred_contact_pts.append(
                        (
                            hand_verts[j, k, vert_ids].view(-1, 3, 3)
                            * bary_coords[..., np.newaxis]
                        ).sum(dim=1)
                    )
            pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(
                self.num_init, -1, 3
            )

            corr_loss = F.mse_loss(pred_contact_pts, object_contact_pts)

            loss = corr_loss
            loss.backward()
            opt.step()

            #§print("Iter {}: {}".format(i, loss.item()))
            #§print("\tCorrespondence Loss: {}".format(corr_loss.item()))
        #print(f"[output] hand_verts: {hand_verts.shape}")

        #print("===============================================================")
        #print("====================== FINE OPTIMIZATION ======================")
        # fine optimization loop
        num_iters = self.num_fine_iter
        opt = torch.optim.Adam(
            [beta_var, rot_var, theta_var, transl_var], lr=self.fine_lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
        for i in range(num_iters):
            opt.zero_grad()
            hand_verts, _ = mano_layer(
                torch.cat([rot_var, theta_var], dim=-1),
                beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10),
                transl_var,
            )
            hand_verts = hand_verts.view(self.num_init, num_frames, 778, 3) #* 0.001

            center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
            hand_verts = torch.cat([hand_verts, center], dim=2)

            shape_prior_loss = torch.mean(beta_var**2)
            pose_prior_loss = torch.mean(theta_var**2)

            pred_contact_pts = []
            for j in range(self.num_init):
                for k in range(num_frames):
                    vert_ids = data_collection[k][0]
                    bary_coords = data_collection[k][1]
                    pred_contact_pts.append(
                        (
                            hand_verts[j, k, vert_ids].view(-1, 3, 3)
                            * bary_coords[..., np.newaxis]
                        ).sum(dim=1)
                    )
            pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(
                self.num_init, -1, 3
            )

            corr_loss = F.mse_loss(pred_contact_pts, object_contact_pts)

            # THESE DO NOT APPLY IN THE SINGLE FRAME SETTING!
            # pose_smoothness_loss = F.mse_loss(
            # theta_var.view(args.num_init, num_frames, -1)[:, 1:],
            # theta_var.view(args.num_init, num_frames, -1)[:, :-1],
            # )
            # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor)

            loss = (
                corr_loss * 20
                # + pose_smoothness_loss * 0.05
                # + joints_smoothness_loss
                + shape_prior_loss * 0.001
                + pose_prior_loss * 0.0001
            )
            loss.backward()
            opt.step()
            scheduler.step()

            #print("Iter {}: {}".format(i, loss.item()), flush=True)
            #print("\tShape Prior Loss: {}".format(shape_prior_loss.item()))
            #print("\tPose Prior Loss: {}".format(pose_prior_loss.item()))
            #print("\tCorrespondence Loss: {}".format(corr_loss.item()))
            # print("\tPose Smoothness Loss: {}".format(pose_smoothness_loss.item()))
            # print("\tJoints Smoothness Loss: {}".format(joints_smoothness_loss.item()))
            assert not torch.isnan(loss).any(), "Loss is NaN"
        #print(f"[output] hand_verts: {hand_verts.shape}")

        #print("===============================================================")
        #print("==================== AGGREGATION/SELECTION ====================")
        # find best initialization
        with torch.no_grad():
            hand_verts, hand_joints = mano_layer(
                torch.cat([rot_var, theta_var], dim=-1),
                beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10),
                transl_var,
            )
            hand_verts = hand_verts.view(self.num_init, num_frames, 778, 3) #* 0.001
            hand_joints = hand_joints.view(self.num_init, num_frames, 21, 3) #* 0.001
            center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
            hand_verts_w_center = torch.cat([hand_verts, center], dim=2)

            pred_contact_pts = []
            #for j in tqdm(range(self.num_init), desc="Selecting best initialization"):
            for j in range(self.num_init):
                for k in range(num_frames):
                    vert_ids = data_collection[k][0]
                    bary_coords = data_collection[k][1]
                    pred_contact_pts.append(
                        (
                            hand_verts_w_center[j, k, vert_ids].view(-1, 3, 3)
                            * bary_coords[..., np.newaxis]
                        ).sum(dim=1)
                    )
            pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(
                self.num_init, -1, 3
            )

            corr_loss = torch.sum(
                (pred_contact_pts - object_contact_pts) ** 2, dim=-1
            ).mean(dim=1)
            min_id = torch.argmin(corr_loss)
            hand_verts = hand_verts[min_id]
            hand_joints = hand_joints[min_id]
        #print(f"[output] hand_verts: {hand_verts.shape}")

        if is_left:
            mano_mesh.f = mano_mesh.f[..., [2, 1, 0]]
            self.mano["f"] = self.mano["f"][..., [2, 1, 0]]

        hand_verts = hand_verts.cpu().numpy()
        hand_joints = hand_joints.cpu().numpy()
        if is_left:
            hand_verts[..., 0] = -hand_verts[..., 0]
            hand_joints[..., 0] = -hand_joints[..., 0]

        assert (hand_verts.shape[0] == 1 and hand_joints.shape[0] == 1), "Output hand vertices should have BS=1."
        #print(f"[output] hand_verts: {hand_verts.shape}")
        #print(f"[output] faces: {mano_mesh.f.shape}")
        #print(f"[input] hand_verts: {input_rhand_pcs[0].shape}")

        if gt is not None:
            ground_truth_hand_mesh = Mesh(v=gt[0].cpu().numpy(), f=mano_mesh.f)
            ground_truth_hand_mesh.write_ply(
                os.path.join("visu", "gt_hand.ply")
            )
			hand_mesh = Mesh(v=hand_verts[0], f=mano_mesh.f)
			hand_mesh_input = seal(Mesh(v=input_rhand_pcs[0], f=self.mano["f"]))
			object_mesh = Mesh(v=object_verts[0], f=obj_mesh.f)
			hand_mesh.write_ply(os.path.join("visu", "hand.ply"))
			hand_mesh_input.write_ply(
				os.path.join("visu", "input_hand.ply")
			)
			object_mesh.write_ply(os.path.join("visu", "object.ply"))

        #print(f"pred hand: {hand_verts[..., :10, :]}, gt hand: {gt[0].cpu().numpy()[..., :10, :]}")
        return {"verts": hand_verts, "faces": mano_mesh.f, "joints": hand_joints}
