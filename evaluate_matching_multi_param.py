from src.evaluation import matching
from src.models import model_loader
import pickle
import numpy as np
import torch
import argparse
import sys
import os
from src.models import SPITE

model_type_to_modalities = {
    "S" : "SKELETON",
    "I" : "IMU",
    "P" : "PC"
}

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Matching Evaluation')
    parser.add_argument('--model_type', default='SPITE', type=str, help='which encoder to use')
    parser.add_argument('--pretrained_path', default="no", type=str, help='which model to use')
    parser.add_argument('--dataset', default="v1", type=str, help='pretrained on which dataset')
    parser.add_argument('--data_path', default=None, type=str, help='path to LIPD_SEQUENCES_256p.pkl')

    parser.add_argument('--n_frames', default=24, type=int, help='random frames')

    #### experimnt args
    parser.add_argument('--num_windows', default=8, type=int, help='Number of artifical subjects')
    parser.add_argument('--window_size', default=4, type=int, help='Temporal window for matching algorithm')
    parser.add_argument('--n_scenes', default=100, type=int, help='Number of artifical scenes, i.e., matching experiments')
    parser.add_argument('--src_modality', default='imu', type=str, help='Source modality that must be matched to one out of N-targets')
    parser.add_argument('--tgt_modality', default='pc', type=str, help='Target modality that serves as matching candidates')
    parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dim of the model')
    parser.add_argument('--skeleton_source', default="gt_joint", choices=["gt_joint", "gt_pose", "gt_pose_joints"], help='Skeleton source to use')
    parser.add_argument('--smpl_backbone', default="transformer", choices=["transformer", "tcn", "conformer", "stgcn"], help='SMPL pose encoder backbone (used when skeleton_source=gt_pose)')
    input_args = parser.parse_args()
    
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data
    print("Loading data.")
    data_path = input_args.data_path
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LIPD", "LIPD_SEQUENCES_256p.pkl")
    print(f"Using data_path: {data_path}")
    with open(data_path, 'rb') as f:
        sequence_datasets = pickle.load(f)
    for m in ["eLIPD", "eTC", "eDIP"]:
        n_subjects = len(sequence_datasets.get(m, {}))
        print(f"Raw {m}: subjects={n_subjects}")

    # Map to uppercase because thats how it is encoded. maybe change in future.
    input_args.src_modality = input_args.src_modality.upper()
    input_args.tgt_modality = input_args.tgt_modality.upper()


    #### Load models
    embed_dim = input_args.embed_dim
    num_joints = 24
    n_feats = 6 if input_args.skeleton_source == "gt_pose" else 3

    # map to modalities based on modeltype
    modalities = [model_type_to_modalities[m].lower() for m in input_args.model_type if not m in ["E", "T"]]
    print(modalities)

    ### Load checkpoint early to infer skeleton input size if needed.
    state_dict = None
    if input_args.pretrained_path != "no":
        ckpt = torch.load(input_args.pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        state_dict = ckpt if isinstance(ckpt, dict) else None

    if "skeleton" in modalities and state_dict:
        skel_keys = [k for k in state_dict.keys() if k.endswith("skelEmbedding.weight") and "skeleton_encoder" in k]
        if skel_keys:
            input_feats = state_dict[skel_keys[0]].shape[1]
            if input_feats % n_feats == 0:
                inferred_joints = input_feats // n_feats
                if inferred_joints != num_joints:
                    print(
                        f"Warning: checkpoint expects {inferred_joints} joints (input_feats={input_feats}), "
                        f"but skeleton_source={input_args.skeleton_source} sets num_joints={num_joints}. "
                        f"Using num_joints={inferred_joints} to match checkpoint."
                    )
                    num_joints = inferred_joints
            else:
                print(
                    f"Warning: checkpoint input_feats={input_feats} not divisible by n_feats={n_feats}; "
                    "using configured num_joints."
                )

    ### Load all models, feed into binder model for training later.
    if "skeleton" in modalities:
        if input_args.skeleton_source == "gt_pose":
            skeleton = model_loader.load_smpl_pose_encoder(
                embed_dim,
                num_joints,
                device="cuda",
                backbone=input_args.smpl_backbone,
            )
        else:
            skeleton = model_loader.load_skeleton_encoder(embed_dim, num_joints, n_feats, device="cuda")
    else:
        skeleton = None
    imu = model_loader.load_imu_encoder(embed_dim, device="cuda") if "imu" in modalities else None
    pc = model_loader.load_pst_transformer(embed_dim, device="cuda") if "pc" in modalities else None
    skeleton_gen = None #model_loader.load_skeleton_generator(embed_dim, num_joints, n_feats, device="cuda") if input_args.with_generator else None

    ### Init the binder model, done.
    model = SPITE.instantiate_binder(modalities, False, imu, pc, skeleton, skeleton_gen).to("cuda")

    if state_dict:
        model.load_state_dict(state_dict, strict=False)

    print("Encoding data.")
    test_subj_datasets = {}
    for m in ["eLIPD", "eTC", "eDIP"]:
        test_subj_dataset = matching.encode_all(sequence_datasets[m], model, window_length=24, model_type=input_args.model_type, skeleton_source=input_args.skeleton_source)
        n_subjects = len(test_subj_dataset)
        n_seqs = sum(len(seqs) for seqs in test_subj_dataset.values())
        if n_seqs == 0:
            print(f"Skipping {m}: no sequences after filtering (skeleton_source={input_args.skeleton_source}).")
            continue
        print(f"{m}: subjects={n_subjects}, sequences={n_seqs}")
        test_subj_datasets[m] = test_subj_dataset
    

    ##### Can just run all kinds of combinates here now.
    
    n_subjects_params = [2, 4, 8, 12, 16, 20, 24, 28, 32] #, 48, 64, 128]
    n_window_sizes_params = [1, 2, 4] #[1, 2, 4, 8] #, 12, 16]
    n_scenes_params = [input_args.n_scenes] # repeat same experiment 10k times

    results_dict = {}
    for n_subjects in n_subjects_params:
        for n_window_size in n_window_sizes_params:
            for n_scenes in n_scenes_params:
                print("")
                print("*******************************")
                print("Starting experiment MODEL=%s - *** %s -> %s ***" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))
                print("#Subjects per scene: %s" % n_subjects)
                print("Temporal matching window: %s" % n_window_size)
                print("#Artifical scenes: %s" % n_scenes)
                print("*******************************")
                print("")
                
                run_results = {}
                #### LidarBind
                for m, test_subj_dataset in test_subj_datasets.items(): 
                    ### Get N Random sequences
                    augmented_scenes = matching.create_augmented_scenes_with_windows(test_subj_dataset, num_windows=n_subjects, 
                                                                                    window_size=n_window_size, n_scenes=n_scenes, 
                                                                                    src_modality=input_args.src_modality, tgt_modality=input_args.tgt_modality)
                    results = matching.eval_scenes(augmented_scenes, src_modality=input_args.src_modality, tgt_modality=input_args.tgt_modality)
                    from sklearn.metrics import accuracy_score
                    avg_acc = []
                    topk_vals = {1: [], 5: [], 10: []}
                    map_vals = []
                    for res in results.values():
                        preds, gt, sims = res
                        avg_acc.append(accuracy_score(preds, gt))

                        # per-scene top-k and mAP
                        correct_at_k = {1: 0, 5: 0, 10: 0}
                        ap_sum = 0.0
                        for qi, sim_list in enumerate(sims):
                            scores = np.array(sim_list)
                            rank = int(np.argsort(-scores).tolist().index(gt[qi])) + 1
                            for k in correct_at_k:
                                kk = min(k, len(scores))
                                if gt[qi] in np.argsort(-scores)[:kk]:
                                    correct_at_k[k] += 1
                            ap_sum += 1.0 / rank

                        n_queries = len(sims)
                        for k in correct_at_k:
                            topk_vals[k].append(correct_at_k[k] / n_queries)
                        map_vals.append(ap_sum / n_queries)

                    print("%s - Average accuracy over all scenes: %s" % (m, np.mean(avg_acc)))
                    print("%s - Top-1: %.4f  Top-5: %.4f  Top-10: %.4f  mAP: %.4f" % (
                        m,
                        np.mean(topk_vals[1]) if topk_vals[1] else 0.0,
                        np.mean(topk_vals[5]) if topk_vals[5] else 0.0,
                        np.mean(topk_vals[10]) if topk_vals[10] else 0.0,
                        np.mean(map_vals) if map_vals else 0.0,
                    ))

                    run_results[m] = {
                        "acc": avg_acc,
                        "topk": topk_vals,
                        "map": map_vals,
                    }  # keep per-scene lists for std/plots
                
                results_dict[(n_subjects, n_window_size, n_scenes)] = run_results

    if not os.path.exists("results/ICCV/matching"):
        os.makedirs("results/ICCV/matching")

    torch.save(results_dict, "results/ICCV/matching/results_matching_%s_%s_%s.pt" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))
