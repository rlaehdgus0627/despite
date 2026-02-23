#!/usr/bin/env python3
import argparse
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

# python 11 fix bug
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


def _iter_sequences(node, prefix=None):
    if prefix is None:
        prefix = []
    if isinstance(node, dict):
        if "gt" in node and "gt_joint" in node:
            yield "/".join(prefix), node
            return
        for k, v in node.items():
            yield from _iter_sequences(v, prefix + [str(k)])


def _select_sequence(dataset: Dict, seq_path: str):
    if seq_path:
        parts = [p for p in seq_path.split("/") if p]
        node = dataset
        for p in parts:
            node = node[p]
        return seq_path, node
    for path, seq in _iter_sequences(dataset):
        return path, seq
    raise RuntimeError("No sequence with gt/gt_joint found in dataset.")


def _load_smplx_model(args, batch_size: int):
    import smplx

    model = smplx.create(
        args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        num_betas=args.num_betas,
        batch_size=batch_size,
    )
    return model.to(args.device)


def _load_hbp_model(args, batch_size: int):
    from human_body_prior.body_model.body_model import BodyModel

    model = BodyModel(args.model_path, num_betas=args.num_betas).to(args.device)
    return model


def _build_pose_tensors(gt: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    gt = gt.reshape(gt.shape[0], -1)
    if gt.shape[1] % 3 != 0:
        raise ValueError(f"gt has {gt.shape[1]} dims; not divisible by 3.")
    njoints = gt.shape[1] // 3
    pose = torch.from_numpy(gt).float().view(-1, njoints, 3)

    # Root orientation is always the first joint.
    root_orient = pose[:, 0:1].contiguous()
    body_pose = pose[:, 1:].contiguous()

    # If we have only 22 joints (root + 21), pad to 24 joints (root + 23).
    if njoints == 22:
        pad = torch.zeros((pose.shape[0], 2, 3), dtype=pose.dtype)
        body_pose = torch.cat([body_pose, pad], dim=1)

    return root_orient, body_pose.view(body_pose.shape[0], -1)


def _select_joints(joints: torch.Tensor, joints_to_use: List[int]) -> torch.Tensor:
    if joints_to_use:
        return joints[:, joints_to_use, :]
    return joints


def main():
    parser = argparse.ArgumentParser(description="Verify if gt matches SMPL pose via joint reconstruction.")
    parser.add_argument(
        "--dataset-pkl",
        default="data/LIPD/LIPD_SEQUENCES_256p.pkl",
        help="Path to LIPD sequences pickle.",
    )
    parser.add_argument(
        "--sequence",
        default="",
        help='Path like "eLIPD/923/seq2". If empty, uses the first found sequence.',
    )
    parser.add_argument("--model-path", required=True, help="Path to SMPL/SMPLH model file or folder.")
    parser.add_argument("--model-type", default="smpl", choices=["smpl", "smplh", "smplx"])
    parser.add_argument("--backend", default="smplx", choices=["smplx", "human_body_prior"])
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument(
        "--joints-to-use",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,37",
        help="Comma-separated joint indices to compare; empty to use all joints.",
    )
    args = parser.parse_args()

    with open(args.dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    seq_path, seq = _select_sequence(dataset, args.sequence)
    gt = np.asarray(seq["gt"])
    gt_joint = np.asarray(seq["gt_joint"])

    # Align frames.
    n_frames = min(gt.shape[0], gt_joint.shape[0], args.max_frames)
    gt = gt[:n_frames]
    gt_joint = gt_joint[:n_frames]

    root_orient, body_pose = _build_pose_tensors(gt)

    if args.backend == "smplx":
        model = _load_smplx_model(args, batch_size=n_frames)
    else:
        model = _load_hbp_model(args, batch_size=n_frames)

    betas = torch.zeros((n_frames, args.num_betas), dtype=torch.float32)
    transl = torch.zeros((n_frames, 3), dtype=torch.float32)

    with torch.no_grad():
        global_orient = root_orient.reshape(root_orient.shape[0], -1)
        body_pose = body_pose.reshape(body_pose.shape[0], -1)
        if args.debug_shapes:
            print(f"root_orient shape: {tuple(root_orient.shape)}")
            print(f"global_orient shape: {tuple(global_orient.shape)}")
            print(f"body_pose shape: {tuple(body_pose.shape)}")
        assert global_orient.ndim == 2, f"global_orient ndim={global_orient.ndim}"
        assert body_pose.ndim == 2, f"body_pose ndim={body_pose.ndim}"
        out = model(
            betas=betas.to(args.device),
            body_pose=body_pose.to(args.device),
            global_orient=global_orient.to(args.device),
            transl=transl.to(args.device),
        )
        joints = out.joints.detach().cpu()

    joints_to_use = []
    if args.joints_to_use:
        joints_to_use = [int(x) for x in args.joints_to_use.split(",") if x.strip()]

    joints = _select_joints(joints, joints_to_use)
    gt_joint_t = torch.from_numpy(gt_joint).float()

    # Compare after trimming to common joint count.
    common_j = min(joints.shape[1], gt_joint_t.shape[1])
    joints = joints[:, :common_j, :]
    gt_joint_t = gt_joint_t[:, :common_j, :]

    err = torch.norm(joints - gt_joint_t, dim=-1).mean().item()
    print(f"sequence: {seq_path}")
    print(f"gt shape: {gt.shape}  gt_joint shape: {gt_joint.shape}")
    print(f"pred joints shape: {joints.shape}")
    print(f"mean joint L2 error: {err:.6f}")


if __name__ == "__main__":
    main()
