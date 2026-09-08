#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import torch


def viz(
    verts: torch.Tensor,
    object: torch.Tensor,
) -> None:
    """
    This visualization (and all visualization functions in this file) is a hell of a mess. I just
    got caught up implementing many different models and I didn't have time to refactor the code.
    Sorry.
    Args:
        model (torch.nn.Module): The model to visualize.
        batch (Union[Tuple, List, torch.Tensor]): The batch to process.
        step (int): The current step.
        bps (torch.Tensor): The basis points set.
        anchor_indices (torch.Tensor): The MANO anchor indices for each basis point.
        bps_dim (int): The dimensionality of the basis points set.
        remap_bps_distances (bool): Whether to remap the basis points set distances to [0, 1].
        dataset (str): The dataset name.
        theta_dim (str): The dimensionality of the MANO pose parameters.
        exponential_map_w (Optional[float], optional): The weight of the exponential map (for remap_bps_distances).
        temporal (bool, optional): Whether the model is temporal or not.
    """
    assert bps_dim == bps.shape[0]
    samples, labels, mesh_paths = batch
    is_baseline = kwargs.get("method", "aggved") == "baseline"
    if not project_conf.HEADLESS:
        # ============ Get the first element of the batch ============
        input_scalar = samples["scalar"][0].view(-1, *samples["scalar"].shape[2:])
        choir_gt = labels["choir"][0, -1].unsqueeze(0)
        input_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        input_ref_pts = samples["rescaled_ref_pts"][0].view(
            -1, *samples["rescaled_ref_pts"].shape[2:]
        )
        gt_scalar = (
            labels["scalar"][0, -1].unsqueeze(0)
            if len(labels["scalar"].shape) >= 2
            else labels["scalar"][-1].unsqueeze(0)
        )
        # gt_scalar = labels["scalar"][0].view(-1, *labels["scalar"].shape[2:])
        gt_ref_pts = labels["rescaled_ref_pts"][0, -1].unsqueeze(0)
        mesh_paths = mesh_paths[-1]  # Now we have a list of B entries.
        N_PTS_ON_MESH = 3000
        obj_mesh = open3d.io.read_triangle_mesh(mesh_paths[0])
        obj_normals = to_cuda_(
            torch.cat(
                (
                    torch.from_numpy(np.asarray(obj_mesh.vertices)).type(
                        dtype=torch.float32
                    ),
                    torch.from_numpy(np.asarray(obj_mesh.vertex_normals)).type(
                        dtype=torch.float32
                    ),
                ),
                dim=-1,
            )
        )
        obj_points = to_cuda_(
            torch.from_numpy(
                np.asarray(obj_mesh.sample_points_uniformly(N_PTS_ON_MESH).points)
            ).float()
        )
        # =============================================================

        if dataset.lower() == "grab":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=24, flat_hand_mean=True, for_contactpose=False)
            )
        elif dataset.lower() == "contactpose":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=15, flat_hand_mean=False, for_contactpose=True)
            )
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset}")
        faces = affine_mano.faces
        F = faces.cpu().numpy()
        # ===================================================================================
        # ============ Display MANO fitted to all noisy input CHOIR fields ==================
        pl = pv.Plotter(
            shape=(1, input_choirs.shape[0] + 1), border=False, off_screen=False
        )
        sample_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        print(f"[*] Visualizing {input_choirs.shape[0]} noisy CHOIR fields.")
        for k, v in samples.items():
            print(f"-> {k}: {v.shape}")
        use_smplx = False  # TODO: for now I'm not interested in using it
        with torch.set_grad_enabled(True):
            (
                _,
                _,
                _,
                _,
                anchors_pred,
                verts_pred,
                joints_pred,
            ) = optimize_pose_pca_from_choir(
                sample_choirs,
                bps=bps,
                anchor_indices=anchor_indices,
                scalar=input_scalar,
                is_rhand=samples["is_rhand"][0],
                # max_iterations=5000,
                max_iterations=1000,
                loss_thresh=1e-6,
                lr=8e-2,
                use_smplx=use_smplx,
                dataset=dataset,
                remap_bps_distances=remap_bps_distances,
                exponential_map_w=exponential_map_w,
                initial_params={
                    k: v[0]
                    for k, v in samples.items()
                    if k
                    in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                },
                beta_w=1e-4,
                theta_w=1e-7,
                choir_w=1000,
                # max_iterations=8000,
                # lr=1e-1,
            )
        for i in range(input_choirs.shape[0]):
            pl.subplot(0, i)
            V = verts_pred[i].cpu().numpy()
            tmesh = Trimesh(V, F)
            hand_mesh = pv.wrap(tmesh)
            add_choir_to_plot(
                pl,
                bps / input_scalar[i],
                input_choirs[i] / input_scalar[i],
                input_ref_pts[i] / input_scalar[i],
                anchors_pred[i],
                hand_mesh,
                hand_only=True,
            )
        # ===================================================================================

        # ================== Optimize MANO on model prediction =========================
        if kwargs.get("method", "aggved") == "baseline":
            conditional = kwargs["conditional"]
            n = 1
            print(
                f"Running with Y={torch.cat( ( samples['rescaled_ref_pts'][0], samples['joints'][0], samples['anchors'][0],), dim=-2,).unsqueeze(0).shape}"
            )
            pred = model.generate(
                n,
                y=torch.cat(
                    (
                        samples["rescaled_ref_pts"][0],
                        samples["joints"][0],
                        samples["anchors"][0],
                    ),
                    dim=-2,
                ).unsqueeze(0)
                if conditional
                else None,
            )
            pred = pred[:, 0]  # TODO: Plot all preds
        elif kwargs.get("method", "aggved") == "ddpm":
            conditional = kwargs["conditional"]
            n = 1
            choir_pred = model.generate(
                n,
                y=samples["choir"][0].unsqueeze(0) if conditional else None,
            )
            choir_pred = choir_pred[:, 0]  # TODO: Plot all preds
        elif kwargs.get("method", "aggved") == "coddpm":
            conditional = kwargs["conditional"]
            n = 1
            choir_pred, contacts_pred = model.generate(
                n,
                y=samples["choir"][0].unsqueeze(0) if conditional else None,
            )
            choir_pred, contacts_pred = choir_pred.squeeze(1), contacts_pred.squeeze(1)
        elif kwargs.get("method", "aggved") == "aggved":
            y_hat = model(samples["choir"], use_mean=True)
            choir_pred = y_hat["choir"][0].unsqueeze(0)
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        mano_params_gt = {k: v[0, -1].unsqueeze(0) for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        verts_gt, joints_gt = affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        anchors_gt = affine_mano.get_anchors(verts_gt)
        with torch.set_grad_enabled(True):
            print(
                "Mean scalar: ",
                torch.mean(input_scalar).unsqueeze(0).to(input_scalar.device),
            )
            if is_baseline:
                joints_pred, anchors_pred = pred[:, :21], pred[:, 21:]
                verts_pred = optimize_mesh_from_joints_and_anchors(
                    pred,
                    scalar=torch.mean(input_scalar)
                    .unsqueeze(0)
                    .to(input_scalar.device),  # TODO: What should I do here?
                    is_rhand=samples["is_rhand"][0],
                    max_iterations=400,
                    loss_thresh=1e-7,
                    lr=8e-2,
                    dataset=dataset,
                    use_smplx=use_smplx,
                    initial_params={
                        k: v[0, -1].unsqueeze(0)
                        for k, v in samples.items()
                        if k
                        in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                    },
                    beta_w=1e-4,
                    theta_w=1e-8,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    anchors_pred,
                    verts_pred,
                    joints_pred,
                ) = optimize_pose_pca_from_choir(
                    choir_pred,
                    contact_gaussians=contacts_pred,
                    obj_pts=obj_points,
                    obj_normals=obj_normals,
                    bps=bps,
                    anchor_indices=anchor_indices,
                    scalar=torch.mean(input_scalar)
                    .unsqueeze(0)
                    .to(input_scalar.device),  # TODO: What should I do here?
                    is_rhand=samples["is_rhand"][0],
                    max_iterations=400,
                    loss_thresh=1e-7,
                    lr=8e-2,
                    # max_iterations=8000,
                    # lr=1e-1,
                    use_smplx=use_smplx,
                    dataset=dataset,
                    remap_bps_distances=remap_bps_distances,
                    exponential_map_w=exponential_map_w,
                    initial_params={
                        k: v[0, -1].unsqueeze(0)
                        for k, v in samples.items()
                        if k
                        in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                    },
                    beta_w=1e-4,
                    theta_w=1e-8,
                    choir_w=1000,
                )
        # ============ Display the ground truth CHOIR field with the GT MANO ================
        V_gt = verts_gt[0].cpu().numpy()
        tmesh_gt = Trimesh(V_gt, F)
        gt_hand_mesh = pv.wrap(tmesh_gt)
        pl.subplot(0, input_choirs.shape[0])
        add_choir_to_plot(
            pl,
            bps / gt_scalar,
            choir_gt / gt_scalar,
            gt_ref_pts / gt_scalar,
            anchors_gt,
            gt_hand_mesh,
            hand_only=True,
            hand_color="red",
        )
        pl.add_title("Noisy input views", font="courier", color="k", font_size=20)
        pl.link_views()
        pl.set_background("white")  # type: ignore
        pl.add_camera_orientation_widget()
        pl.show(interactive=True)

        # ============ Display the optimized MANO on predicted CHOIR field vs GT CHOIR field ================
        # ====== Metrics and qualitative comparison ======
        # === Anchor error ===
        print(
            f"Anchor error (mm): {torch.norm(anchors_pred - anchors_gt.to(anchors_pred.device), dim=2).mean(dim=1).mean(dim=0).item() * 1000:.2f}"
        )
        # === MPJPE ===
        pjpe = torch.linalg.vector_norm(
            joints_gt.to(joints_pred.device) - joints_pred, ord=2, dim=-1
        )  # Per-joint position error (B, N, 21)
        mpjpe = torch.mean(pjpe, dim=-1).item()  # Mean per-joint position error (B, N)
        print(f"MPJPE (mm): {mpjpe * 1000:.2f}")
        root_aligned_pjpe = torch.linalg.vector_norm(
            (joints_gt - joints_gt[:, 0, :]).to(joints_pred.device)
            - (joints_pred - joints_pred[:, 0, :]),
            ord=2,
            dim=-1,
        )  # Per-joint position error (B, N, 21)
        root_aligned_mpjpe = torch.mean(
            root_aligned_pjpe, dim=-1
        ).item()  # Mean per-joint position error (B, N)
        print(f"Root-aligned MPJPE (mm): {root_aligned_mpjpe * 1000:.2f}")
        # ====== MPVPE ======
        # Compute the mean per-vertex position error (MPVPE) between the predicted and ground truth
        # hand meshes.
        pvpe = torch.linalg.vector_norm(
            verts_gt - verts_pred, ord=2, dim=-1
        )  # Per-vertex position error (B, N, 778)
        mpvpe = torch.mean(pvpe, dim=-1).item()  # Mean per-vertex position error (B, N)
        print(f"MPVPE (mm): {mpvpe * 1000:.2f}")

        V = verts_pred[0].cpu().numpy()
        tmesh = Trimesh(V, F)
        # hand_mesh = pv.wrap(tmesh)

        visualize_MANO(
            tmesh, obj_ptcld=input_ref_pts[0] / input_scalar[0], gt_hand=gt_hand_mesh
        )
