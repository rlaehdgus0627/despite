#!/usr/bin/env python3
import argparse
import pickle
from typing import Dict, Iterable, Tuple

import torch
import smplx


def _iter_sequences(node, prefix=None):
    if prefix is None:
        prefix = []
    if isinstance(node, dict):
        if "gt" in node:
            yield "/".join(prefix), node
            return
        for k, v in node.items():
            yield from _iter_sequences(v, prefix + [str(k)])


def _select_datasets(data: Dict, datasets: Iterable[str]):
    if not datasets:
        return data.keys()
    return [d for d in datasets if d in data]


def _pose_to_joints(model, gt_pose: torch.Tensor, num_betas: int, batch_size: int):
    # gt_pose: (T, 72) axis-angle
    T = gt_pose.shape[0]
    njoints = gt_pose.shape[1] // 3
    pose = gt_pose.view(T, njoints, 3)
    root_orient = pose[:, 0:1].contiguous()
    body_pose = pose[:, 1:].contiguous()
    if njoints == 22:
        pad = torch.zeros((T, 2, 3), dtype=pose.dtype)
        body_pose = torch.cat([body_pose, pad], dim=1)

    joints_out = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        n = end - start
        betas = torch.zeros((n, num_betas), dtype=torch.float32)
        transl = torch.zeros((n, 3), dtype=torch.float32)
        global_orient = root_orient[start:end].reshape(n, -1)
        body_pose_flat = body_pose[start:end].reshape(n, -1)
        with torch.no_grad():
            out = model(
                betas=betas,
                body_pose=body_pose_flat,
                global_orient=global_orient,
                transl=transl,
            )
            joints = out.joints.detach().cpu()
        joints_out.append(joints)

    return torch.cat(joints_out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Convert gt (axis-angle) to SMPL joints and store as gt_pose_joints.")
    parser.add_argument("--input-pkl", default="data/LIPD/LIPD_SEQUENCES_256p.pkl")
    parser.add_argument("--output-pkl", default="data/LIPD/LIPD_SEQUENCES_256p_posejoints.pkl")
    parser.add_argument("--model-path", required=True, help="Path to SMPL/SMPLH/SMPLX model file or folder.")
    parser.add_argument("--model-type", default="smpl", choices=["smpl", "smplh", "smplx"])
    parser.add_argument("--gender", default="male")
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--datasets", nargs="*", default=["eTC"], help="Datasets to process, e.g. eTC eDIP eLIPD")
    args = parser.parse_args()

    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    model = smplx.create(
        args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        num_betas=args.num_betas,
        batch_size=args.batch_size,
    )

    for ds in _select_datasets(data, args.datasets):
        for _, seq in _iter_sequences(data[ds], prefix=[ds]):
            gt = seq.get("gt")
            if gt is None:
                continue
            gt_pose = torch.as_tensor(gt).float().view(gt.shape[0], -1)
            joints = _pose_to_joints(model, gt_pose, args.num_betas, args.batch_size)
            # keep 24 joints to match existing pipeline
            if joints.shape[1] > 24:
                joints = joints[:, :24, :]
            seq["gt_pose_joints"] = joints

    with open(args.output_pkl, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
