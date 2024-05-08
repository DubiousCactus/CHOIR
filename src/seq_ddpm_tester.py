#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Motion Sequence DDPM Tester.
"""


import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh

from src.multiview_tester import MultiViewTester
from utils import to_cuda
from utils.anim import ScenePicAnim
from utils.testing import make_batch_of_obj_data, mp_process_obj_meshes
from utils.training import optimize_pose_pca_from_choir


@dataclass
class CacheEntry:
    frame_count: int
    marked_for_optimization: bool
    mano: Dict
    verts: torch.Tensor


class SeqDDPMTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._data_loader.dataset.set_observations_number(1)
        self._full_choir = kwargs.get("full_choir", False)
        self._use_deltas = self._data_loader.dataset.use_deltas
        self._model_contacts = kwargs.get("model_contacts", False)
        self._enable_contacts_tto = kwargs.get("enable_contacts_tto", False)
        self._use_ema = kwargs.get("use_ema", False)
        self._n_augmentations = 10
        self._scenes_cache = {}
        self._window_size = 10
        if self._model_contacts:
            self._model.backbone.set_anchor_indices(
                self._data_loader.dataset.anchor_indices
            )
            self._model.set_dataset_stats(self._data_loader.dataset)
            if self._ema is not None:
                self._ema.ema_model.backbone.set_anchor_indices(
                    self._data_loader.dataset.anchor_indices
                )
                self._ema.ema_model.set_dataset_stats(self._data_loader.dataset)
        self._single_modality = self._model.single_modality
        # Because I infer the shape of the model from the data, I need to
        # run the model's forward pass once before calling .generate()
        if kwargs.get("compile_test_model", False):
            print("[*] Compiling the model...")
            self._model = torch.compile(self._model)
            if self._ema is not None:
                self._ema.ema_model = torch.compile(self._ema.ema_model)

    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        max_observations = max_observations or samples["choir"].shape[1]
        model = self._ema.ema_model if self._use_ema else self._model
        if self._single_modality is not None:
            modality = self._single_modality
        else:
            modality = "noisy_pair" if self._inference_mode == "denoising" else "object"
        print(
            f"y shape: {samples['choir'].shape}, max_observations: {max_observations}"
        )
        y = (
            samples["choir"][:, :1] if self.conditional else None
        )  # We're using only one frame
        print(f"[*] Using modality: {modality}")
        if modality == "object":
            y = y[..., 0].unsqueeze(-1)
        elif modality == "noisy_pair":
            # Already comes in noisy_pair modality
            pass
        indices = kwargs.get("indices", None)
        if indices is not None:
            y = y[indices]
        udf, contacts = model.generate(
            1,
            y=y,
            y_modality=modality,
        )
        rotations = None
        # Only use 1 sample for now. TODO: use more samples and average?
        # No, I will do what GraspTTA is doing: apply N random rotations to the object and draw one
        # sample. It's the fairest way to compare since GraspTTA can't get any grasp variations
        # from just sampling z (unlike ours hehehe).
        # TODO: Apply augmentations to the test dataset (can't rotate CHOIRs easily).
        udf, contacts = udf.squeeze(1), contacts.squeeze(1)
        # If I passed indices, I still want to return the same shape as y would have without
        # indices:
        _udf = (
            torch.zeros_like(udf)
            .expand(samples["choir"].shape[0], *udf.shape[1:])
            .clone()
        )
        _udf[indices] = udf
        _contacts = (
            torch.zeros_like(contacts)
            .expand(_udf.shape[0], *contacts.shape[1:])
            .clone()
        )
        _contacts[indices] = contacts
        udf, contacts = _udf, _contacts
        return {"choir": udf, "contacts": contacts, "rotations": rotations}

    @to_cuda
    def _save_batch_predictions_as_sequence(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        sample_pths: List[str],
        n_observations: int,
        batch_idx: int,
        scenes: Dict,
    ):
        """
        This function takes in a batch of sequential frames and a scenes dictionary, and returns
        that dictionary with the new frames added to it..
        """
        mesh_pths = list(mesh_pths[-1])  # Now we have a list of B entries.
        sample_pths = list(sample_pths[-1])  # Now we have a list of B entries.
        mp_process_obj_meshes(
            mesh_pths,
            self._object_cache,
            self._data_loader.dataset.center_on_object_com,
            self._enable_contacts_tto,
            self._compute_iv,
            self._pitch,
            self._radius,
            self._n_pts_on_mesh,
            self._n_normals_on_mesh,
            dataset=self._data_loader.dataset.name,
            keep_mesh_contact_identity=False,
        )
        batch_obj_data = make_batch_of_obj_data(
            self._object_cache, mesh_pths, keep_mesh_contact_identity=False
        )
        for i in range(len(batch_obj_data["mesh_name"])):
            mesh_name = batch_obj_data["mesh_name"][i].split(".")[0]
            if mesh_name in ["wineglass"]:
                # Remove all batch elements before the first frame that has the object we want.
                batch_obj_data = {k: v[i:] for k, v in batch_obj_data.items()}
                sample_pths = sample_pths[i:]
                mesh_pths = mesh_pths[i:]
                break
        else:  # Didn't find the right mesh
            return scenes
        print(
            "Rendering: "
            + ", ".join(
                list(
                    set(
                        [
                            mesh_name.split(".")[0]
                            for mesh_name in batch_obj_data["mesh_name"]
                        ]
                    )
                )
            )
        )
        hand_color = trimesh.visual.random_color()
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2:
            input_scalar = input_scalar.mean(
                dim=1
            )  # TODO: Think of a better way for 'pair' scaling. Never mind we have object scaling which is better
        mano_params_gt = {
            "pose": labels["theta"].view(-1, *labels["theta"].shape[2:]),
            "beta": labels["beta"].view(-1, *labels["beta"].shape[2:]),
            "rot_6d": labels["rot"].view(-1, *labels["rot"].shape[2:]),
            "trans": labels["trans"].view(-1, *labels["trans"].shape[2:]),
        }
        mano_params_input = {
            "pose": samples["theta"].view(-1, *samples["theta"].shape[2:]),
            "beta": samples["beta"].view(-1, *samples["beta"].shape[2:]),
            "rot_6d": samples["rot"].view(-1, *samples["rot"].shape[2:]),
            "trans": samples["trans"].view(-1, *samples["trans"].shape[2:]),
        }
        # Only use the last view for each batch element (they're all the same anyway for static
        # grasps, but for dynamic grasps we want to predict the LAST frame!).
        # mano_params_gt = {k: v for k, v in mano_params_gt.items()}
        gt_verts, _ = self._affine_mano(
            mano_params_gt["pose"],
            mano_params_gt["beta"],
            mano_params_gt["trans"],
            rot_6d=mano_params_gt["rot_6d"],
        )
        sample_verts, sample_joints = self._affine_mano(
            mano_params_input["pose"],
            mano_params_input["beta"],
            mano_params_input["trans"],
            rot_6d=mano_params_input["rot_6d"],
        )
        if not self._data_loader.dataset.is_right_hand_only:
            raise NotImplementedError("Right hand only is implemented for testing.")
        multiple_obs = len(samples["theta"].shape) > 2
        # For mesh_pths we have a tuple of N lists of B entries. N is the number of
        # observations and B is the batch size. We'll take the last observation for each batch
        # element.
        use_smplx = False  # TODO: I don't use it for now

        """
        Ok let's implement the actual logic here. We have a cache where we can store the
        previous frame's fitted MANO parameters for the given scenes in the batch. We will use
        them as initial parameters for the optimization, unless it's not the cache in which case
        we'll use the noisy input parameters. Then we replace the cache entry with the optimized
        parameters.
        We need to adapt the optimize_pose_pca_from_choir method so that it returns a
        sequence of MANO frames. From this sequence we'll sample N frames using np.logspace(0, N)
        to accomodate for the fact that convergence speed is like a logarithmic curve.

        1. Find the scene in the cache, retrieve the MANO parameters.
        2. Run the optimization with the MANO parameters as initial parameters.
        3. Store the optimized MANO parameters in the cache.
        """
        scene_names = [pathlib.PurePath(path).parent.name for path in sample_pths]
        # TODO: What happens if we have multiple scenes in the batch? Can we handle more than 3?
        # Well, we work with sequences. So in the batch we either have B frames of the same scene,
        # or B frames of different scenes. We want to run optimization of ONE frame per scene,
        # since we interpolate between N frames. If we have more than 1 scene in the batch, we need
        # to run batch optimization with K initializations for K scenes, and retrieve the
        # corresponding cached result for N_0 for each scene of the K scenes.

        # Find the indices in the scene_names array where the scene changes:
        # TODO: Refactor the following two list comprehensions into one simple loop.
        new_scene_indices = [
            i
            for i in range(len(scene_names))
            if scene_names[i] != scene_names[max(i - 1, 0)]
            or scene_names[i] not in self._scenes_cache
        ]
        # Prune new scenes that are duplicates:
        new_scene_indices = [
            i for i in new_scene_indices if scene_names[i] not in scene_names[:i]
        ]
        individual_scenes = set(scene_names)

        # next we need to find the frame to optimize for if it's in the batch. It must abide
        # one of the following conditions: A. It's the last frame of the sequence, B. It's the
        # frame number (using the running counter in the cache entry) that is equal to the window size.
        # A. Find the last frame of the sequence(s).
        last_frames = []
        for i in range(len(scene_names)):
            # TODO: Interpolate with N - frame_count frames not N frames
            if i not in new_scene_indices:
                continue
            last_frame_idx = max(i - 1, 0)
            if last_frame_idx not in new_scene_indices:
                last_frames.append(last_frame_idx)

        # B. Find the frames due for optimization (window full).
        # 1. I want to find the first frame that brings the counter to the window size, if it's in
        # the batch.
        # 2. I want to update the frame counter, taking 1. into account (meaning overflow/reset if
        # needed).

        # For each inidividual scene, look up the cache entry or init to the noisy input.
        # Then build a batch of initial MANO paramters from these, and run the optimization to get
        # a sequence of N=self._window_size frames for each scene. Then update the cache.
        due_frames = []
        for i in range(len(scene_names)):
            if scene_names[i] not in self._scenes_cache:
                continue
            # For each frame in the batch, update the cache entry's frame count.
            self._scenes_cache[scene_names[i]].frame_count += 1
            # If the frame count has reached window_size, use this corresponding frame as the
            # optimization target.
            # If we matched, reset the frame count to 0 and keep counting the following
            # frames. If we matched more than once per cache entry, the batch size is too large and
            # we can't handle this case for now (abort and warn, this is not needed). We can only
            # match once per batch because optimization of frames depends on the solution of
            # previous frames.
            if self._scenes_cache[scene_names[i]].frame_count >= self._window_size:
                if self._scenes_cache[scene_names[i]].marked_for_optimization:
                    raise NotImplementedError(
                        "Can't handle more than one window per scene in the batch. Please reduce the batch size (try 8)."
                    )
                due_frames.append(i)
                self._scenes_cache[scene_names[i]].frame_count = 0
                self._scenes_cache[scene_names[i]].marked_for_optimization = True

        # ========================================================================================
        frames_to_optimize = {
            "choir": [],
            "contacts": [],
            "obj_pts": [],
            "obj_normals": [],
            "scalar": [],
            "is_rhand": [],
            "init_params": [],
            "obj_mesh": [],
            "mesh_name": [],
            "scene_name": [],
        }
        frame_indices = set(last_frames + due_frames)
        y_hat = (
            None
            if len(frame_indices) == 0
            else self._inference(
                samples, labels, use_prior=True, indices=list(frame_indices)
            )  # Pass indices to run inference on just on the frames we need to optimize.
        )
        for i in frame_indices:
            contacts = y_hat.get("contacts", None)
            if i not in new_scene_indices:
                if scene_names[i] not in self._scenes_cache:
                    raise
                assert (
                    scene_names[i] in self._scenes_cache
                ), f"Scene {scene_names[i]} not in cache, but {scene_names[i]} is not a new scene!"
                initial_params = self._scenes_cache[scene_names[i]].mano
                self._scenes_cache[scene_names[i]].marked_for_optimization = False
            else:
                initial_params = {
                    k: (
                        v[i, -1] if multiple_obs else v[i]
                    )  # Initial pose is the last observation
                    for k, v in samples.items()
                    if k
                    in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                }
            assert (
                initial_params is not None
            ), f"Initial params are None for {scene_names[i]}"
            frames_to_optimize["choir"].append(y_hat["choir"][i])
            frames_to_optimize["contacts"].append(
                contacts[i] if contacts is not None else None
            )
            frames_to_optimize["obj_pts"].append(batch_obj_data["points"][i])
            frames_to_optimize["obj_normals"].append(batch_obj_data["normals"][i])
            frames_to_optimize["scalar"].append(input_scalar[i])
            frames_to_optimize["is_rhand"].append(samples["is_rhand"][i])
            frames_to_optimize["init_params"].append(initial_params)
            frames_to_optimize["obj_mesh"].append(batch_obj_data["mesh"][i])
            frames_to_optimize["mesh_name"].append(
                batch_obj_data["mesh_name"][i].split(".")[0]
            )
            frames_to_optimize["scene_name"].append(scene_names[i])

        # It's perfectly ok to not have any frames to optimize, because we may have a small batch
        # and we have to wait to get self._window_size frames. Meaning we need to keep track of
        # scene frames in the cache and only optimize when we have enough frames.
        mano_sequence_pred, verts_sequence_pred = [], []
        if len(frames_to_optimize["choir"]) > 0:
            # Make tensors from lists of tensors:
            for k, v in frames_to_optimize.items():
                if v != [] and isinstance(v[0], torch.Tensor):
                    frames_to_optimize[k] = torch.stack(v)
            # Coalesce the list of init param dicts into a single dict:
            frames_to_optimize["init_params"] = {
                k: torch.stack([v[k] for v in frames_to_optimize["init_params"]])
                for k in frames_to_optimize["init_params"][0].keys()
            }
            with torch.set_grad_enabled(True):
                (
                    _,
                    _,
                    _,
                    _,
                    mano_sequence_pred,
                    verts_sequence_pred,
                ) = optimize_pose_pca_from_choir(
                    frames_to_optimize["choir"],
                    # TODO: Handle the frames and convergence discrepency when enabling contacts
                    # TTO
                    contact_gaussians=frames_to_optimize["contacts"]
                    if self._enable_contacts_tto
                    else None,
                    bps=self._bps,
                    obj_pts=frames_to_optimize["obj_pts"],
                    obj_normals=frames_to_optimize["obj_normals"],
                    anchor_indices=self._anchor_indices,
                    scalar=frames_to_optimize["scalar"],
                    max_iterations=3000,
                    loss_thresh=1e-6,
                    lr=1e-3,
                    lr_stepsize=50,
                    lr_gamma=0.9,
                    is_rhand=frames_to_optimize["is_rhand"],
                    use_smplx=use_smplx,
                    dataset=self._data_loader.dataset.name,
                    remap_bps_distances=self._remap_bps_distances,
                    exponential_map_w=self._exponential_map_w,
                    initial_params=frames_to_optimize["init_params"],
                    beta_w=1e-4,
                    theta_w=1e-7,
                    choir_w=1000,
                    obj_meshes=None,
                    save_tto_anim=False,
                    return_sequence=True,
                )
            # Store the optimized MANO parameters in the cache:
            for i in frame_indices:
                self._scenes_cache[scene_names[i]].marked_for_optimization = False
                self._scenes_cache[scene_names[i]].mano = mano_sequence_pred[-1]
                self._scenes_cache[scene_names[i]].verts = verts_sequence_pred[-1]

        # We also need to add the initial noisy input to the cache entries if it's a new sequence:
        for i in new_scene_indices:
            assert scene_names[i] not in self._scenes_cache
            mano_params = {
                k: (
                    v[:, i] if multiple_obs else v[i]
                )  # Initial pose is the last observation
                for k, v in samples.items()
                if k in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
            }
            verts = sample_verts[i]
            self._scenes_cache[scene_names[i]] = CacheEntry(
                0, False, mano_params, verts
            )

        # TODO: See how to properly sample when contact fitting is enabled. A first simple strategy
        # is to concatenate frames of both stages and still use np.geomspace(0, N) sampling.
        # TODO: What I have now may give me what I want, but the ground-truth and input hand meshes
        # aren't right because I skip a lot of batches and drop them until I have a full window.
        # Instead, I need to cache them for when I'm ready to dump processed preds.
        if len(verts_sequence_pred) < self._window_size:
            return scenes
        sampling_indices = (
            np.geomspace(1, len(verts_sequence_pred), self._window_size, endpoint=False)
            .round()
            .astype(int)
        )
        sampling_indices -= np.ones_like(
            sampling_indices
        )  # Because np.geomspace can't start at 0
        for j in range(len(list(frames_to_optimize.values())[0])):
            for i, (mano_params, verts) in enumerate(
                zip(mano_sequence_pred, verts_sequence_pred)
            ):
                if i not in sampling_indices:
                    continue
                # verts has batch dimension that corresponds to the frames to optimize.
                pred_hand_mesh = trimesh.Trimesh(
                    vertices=verts[j],
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                pred_hand_mesh.visual.vertex_colors = hand_color
                # i corresponds to batch element
                # sample_verts is (B, T, V, 3) but (B*T, V, 3) actually. So to index [i, n-1] we
                # need to do [i * T + n - 1]. n-1 because n is 1-indexed.
                input_hand_mesh = trimesh.Trimesh(
                    vertices=sample_verts[
                        j * samples["theta"].shape[1] + n_observations - 1
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                gt_hand_mesh = trimesh.Trimesh(
                    vertices=gt_verts[j * labels["theta"].shape[1] + n_observations - 1]
                    .detach()
                    .cpu()
                    .numpy(),
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                key = frames_to_optimize["scene_name"][j]
                if key not in scenes:
                    scene_anim = ScenePicAnim()
                    scenes[key] = scene_anim

                scenes[key].add_frame(
                    {
                        "object": frames_to_optimize["obj_mesh"][j],
                        "hand": pred_hand_mesh,
                        "hand_aug": input_hand_mesh,
                        "gt_hand": gt_hand_mesh,
                    }
                )
        return scenes
