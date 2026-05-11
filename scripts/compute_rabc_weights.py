#!/usr/bin/env python

# Batched version of lerobot's compute_rabc_weights.py.
# All CLI parameters are identical; --batch-size is added to control GPU batch size.
# The only behavioral change: reward model inference is batched (default 32) instead
# of one frame at a time, which fully utilises H100/A100 GPU throughput.

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel
from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors
from lerobot.rewards.sarm.sarm_utils import normalize_stage_tau


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def load_sarm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    device: str = "cuda",
) -> tuple[LeRobotDataset, SARMRewardModel, any]:
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices

    logging.info(f"Loading dataset: {dataset_repo_id}")
    temp_dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    fps = temp_dataset.fps

    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    return dataset, reward_model, preprocess


def to_numpy_image(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    ax_frames.axis("off")
    num_sample = 8
    sample_indices = np.linspace(0, len(frames) - 1, num_sample, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[idx])][:12]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _run_batched_inference(
    reward_model: SARMRewardModel,
    processed_list: list[dict],
    device: str,
    compute_sparse: bool,
    compute_dense: bool,
    center_idx: int,
    head_mode_sparse: str = "sparse",
    head_mode_dense: str = "dense",
    return_stages: bool = False,
) -> dict[int, tuple]:
    """
    Run a single batched forward pass over a list of preprocessed samples.

    Each entry in processed_list is a dict of tensors with batch dim = 1.
    Returns a dict mapping list-position -> (sparse_val, dense_val) or
    list-position -> (sparse_reward, sparse_stages) when return_stages=True.
    """
    video_batch = torch.cat([p["video_features"] for p in processed_list], dim=0).to(device)
    text_batch = torch.cat([p["text_features"] for p in processed_list], dim=0).to(device)

    state_list = [p.get("state_features") for p in processed_list]
    state_batch = (
        torch.cat(state_list, dim=0).to(device)
        if all(s is not None for s in state_list)
        else None
    )

    lengths_list = [p.get("lengths") for p in processed_list]
    lengths_batch = (
        torch.cat(lengths_list, dim=0).to(device)
        if all(l is not None for l in lengths_list)
        else None
    )

    results = {}
    with torch.no_grad():
        if return_stages:
            # Used by visualize path — returns (reward [B,T], stages [B,T,S])
            reward, stage_probs = reward_model.calculate_rewards(
                text_embeddings=text_batch,
                video_embeddings=video_batch,
                state_features=state_batch,
                lengths=lengths_batch,
                return_all_frames=True,
                return_stages=True,
                head_mode=head_mode_sparse,
            )
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
                stage_probs = stage_probs.cpu().numpy()
            for i in range(len(processed_list)):
                if reward.ndim == 2:
                    results[i] = (reward[i, center_idx], stage_probs[i, center_idx, :])
                else:
                    results[i] = (reward[center_idx], stage_probs[center_idx, :])
        else:
            sparse_val_batch = None
            dense_val_batch = None

            if compute_sparse:
                sparse_out = reward_model.calculate_rewards(
                    text_embeddings=text_batch,
                    video_embeddings=video_batch,
                    state_features=state_batch,
                    lengths=lengths_batch,
                    return_all_frames=True,
                    head_mode=head_mode_sparse,
                )
                if sparse_out.ndim == 2:
                    sparse_val_batch = sparse_out[:, center_idx].cpu().numpy()
                else:
                    sparse_val_batch = sparse_out[center_idx].cpu().numpy()

            if compute_dense:
                dense_out = reward_model.calculate_rewards(
                    text_embeddings=text_batch,
                    video_embeddings=video_batch,
                    state_features=state_batch,
                    lengths=lengths_batch,
                    return_all_frames=True,
                    head_mode=head_mode_dense,
                )
                if dense_out.ndim == 2:
                    dense_val_batch = dense_out[:, center_idx].cpu().numpy()
                else:
                    dense_val_batch = dense_out[center_idx].cpu().numpy()

            for i in range(len(processed_list)):
                sv = float(sparse_val_batch[i]) if sparse_val_batch is not None else np.nan
                dv = float(dense_val_batch[i]) if dense_val_batch is not None else np.nan
                results[i] = (sv, dv)

    return results


def visualize_sarm_predictions(
    dataset: LeRobotDataset,
    reward_model: SARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: Path,
    num_display_frames: int = 5,
    stride: int = 1,
    batch_size: int = 32,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    dual_mode = reward_model.config.uses_dual_heads
    device = reward_model.device

    target_idx = reward_model.config.n_obs_steps // 2

    schemes_to_viz = []
    if head_mode in ("sparse", "both") or not dual_mode:
        schemes_to_viz.append("sparse")
    if head_mode in ("dense", "both") and dual_mode:
        schemes_to_viz.append("dense")

    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        task = dataset[ep_start].get("task", "perform the task")
        num_frames = ep_end - ep_start

        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames
            else list(range(ep_start, ep_end))
        )
        viz_frames = {}
        for frame_idx in display_indices:
            sample = dataset[frame_idx]
            viz_frames[frame_idx] = to_numpy_image(sample[image_key])

        scheme_data = {}
        for scheme in schemes_to_viz:
            num_stages = getattr(reward_model.config, f"num_{scheme}_stages")
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan),
                "viz_stages": np.full((num_frames, num_stages), np.nan),
                "viz_gt_progress": np.full(num_frames, np.nan),
                "viz_gt_stages": np.full(num_frames, np.nan),
                "target_key": f"{scheme}_targets",
                "num_stages": num_stages,
                "temporal_props": getattr(reward_model.config, f"{scheme}_temporal_proportions"),
                "subtask_names": getattr(reward_model.config, f"{scheme}_subtask_names"),
            }

        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        # Batch the visualization inference
        buffer_frame_idxs = []
        buffer_processed = []
        buffer_local_idxs = []

        def _flush_viz_buffer():
            if not buffer_processed:
                return
            for scheme in schemes_to_viz:
                batch_results = _run_batched_inference(
                    reward_model=reward_model,
                    processed_list=buffer_processed,
                    device=device,
                    compute_sparse=(scheme == "sparse"),
                    compute_dense=(scheme == "dense"),
                    center_idx=target_idx,
                    return_stages=True,
                )
                sd = scheme_data[scheme]
                for i, local_idx in enumerate(buffer_local_idxs):
                    reward_val, stage_val = batch_results[i]
                    sd["viz_progress"][local_idx] = reward_val
                    sd["viz_stages"][local_idx] = stage_val

                    if stride == 1 and sd["target_key"] in buffer_processed[i]:
                        gt_target = buffer_processed[i][sd["target_key"]][0, target_idx].cpu().item()
                        sd["viz_gt_stages"][local_idx] = int(gt_target)
                        sd["viz_gt_progress"][local_idx] = normalize_stage_tau(
                            gt_target,
                            num_stages=sd["num_stages"],
                            temporal_proportions=sd["temporal_props"],
                            subtask_names=sd["subtask_names"],
                        )
            buffer_frame_idxs.clear()
            buffer_processed.clear()
            buffer_local_idxs.clear()

        for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                image_key: sample[image_key],
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)

            buffer_frame_idxs.append(frame_idx)
            buffer_processed.append(processed)
            buffer_local_idxs.append(local_idx)

            if len(buffer_processed) >= batch_size:
                _flush_viz_buffer()

        _flush_viz_buffer()

        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]
                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(
                        valid_idx, sd["viz_progress"][valid_idx], all_local
                    )
                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for s in range(sd["num_stages"]):
                        stage_interp[:, s] = interpolate_progress(
                            valid_idx, sd["viz_stages"][valid_idx, s], all_local
                        )
                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    nz = row_sums.squeeze(-1) > 0
                    stage_interp[nz] = stage_interp[nz] / row_sums[nz]
                    sd["viz_stages"] = stage_interp
                else:
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        ordered_viz_frames = [viz_frames[idx] for idx in sorted(display_indices)]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            viz_path = output_dir / f"sarm_prediction_ep{episode_idx}_{scheme}.png"
            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx})",
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None,
                gt_stages=sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None,
            )

        torch.cuda.empty_cache()

    logging.info(f"Visualizations saved to: {output_dir.absolute()}")


def generate_all_frame_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    num_frames = ep_end - ep_start
    indices = []
    for offset in range(frame_gap):
        for frame_rel in range(offset, num_frames, frame_gap):
            indices.append(ep_start + frame_rel)
    return indices


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    num_visualizations: int = 5,
    output_dir: str = "./sarm_viz",
    stride: int = 1,
    batch_size: int = 32,
):
    dataset, reward_model, preprocess = load_sarm_resources(dataset_repo_id, reward_model_path, device)

    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    logging.info(f"Processing {total_frames} frames across {num_episodes} episodes (batch_size={batch_size})")

    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_progress_sparse = [] if compute_sparse else None
    all_progress_dense = [] if compute_dense else None

    center_idx = reward_model.config.n_obs_steps // 2

    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        task = dataset[ep_start].get("task", "perform the task")

        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        frame_results = {}

        # --- batched inference ---
        buffer_query_idxs: list[int] = []
        buffer_processed: list[dict] = []

        def _flush_batch():
            if not buffer_processed:
                return
            try:
                batch_results = _run_batched_inference(
                    reward_model=reward_model,
                    processed_list=buffer_processed,
                    device=device,
                    compute_sparse=compute_sparse,
                    compute_dense=compute_dense,
                    center_idx=center_idx,
                )
                for i, q_idx in enumerate(buffer_query_idxs):
                    frame_results[q_idx] = batch_results[i]
            except Exception as e:
                logging.warning(f"Batch failed, falling back to per-frame: {e}")
                for i, q_idx in enumerate(buffer_query_idxs):
                    try:
                        sv, dv = np.nan, np.nan
                        p = buffer_processed[i]
                        vf = p["video_features"].to(device)
                        tf = p["text_features"].to(device)
                        sf = p.get("state_features")
                        if sf is not None:
                            sf = sf.to(device)
                        lg = p.get("lengths")
                        with torch.no_grad():
                            if compute_sparse:
                                out = reward_model.calculate_rewards(
                                    text_embeddings=tf,
                                    video_embeddings=vf,
                                    state_features=sf,
                                    lengths=lg,
                                    return_all_frames=True,
                                    head_mode="sparse",
                                )
                                sv = float(out[0, center_idx] if out.ndim == 2 else out[center_idx])
                            if compute_dense:
                                out = reward_model.calculate_rewards(
                                    text_embeddings=tf,
                                    video_embeddings=vf,
                                    state_features=sf,
                                    lengths=lg,
                                    return_all_frames=True,
                                    head_mode="dense",
                                )
                                dv = float(out[0, center_idx] if out.ndim == 2 else out[center_idx])
                        frame_results[q_idx] = (sv, dv)
                    except Exception as e2:
                        logging.warning(f"Failed frame {q_idx}: {e2}")
            finally:
                buffer_query_idxs.clear()
                buffer_processed.clear()

        for query_idx in tqdm(compute_indices, desc=f"  Ep {episode_idx}", leave=False):
            try:
                sample = dataset[query_idx]
                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]

                with torch.no_grad():
                    processed = preprocess(batch)

                buffer_query_idxs.append(query_idx)
                buffer_processed.append(processed)

                if len(buffer_processed) >= batch_size:
                    _flush_batch()

            except Exception as e:
                logging.warning(f"Failed to load frame {query_idx}: {e}")

        _flush_batch()  # remainder
        # -------------------------

        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            all_indices.append(frame_idx)
            all_episode_indices.append(episode_idx)
            all_frame_indices.append(local_idx)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    all_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    all_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    all_progress_dense.append(frame_results[frame_idx][1])
                else:
                    all_progress_dense.append(np.nan)

        torch.cuda.empty_cache()

    table_data = {
        "index": np.array(all_indices, dtype=np.int64),
        "episode_index": np.array(all_episode_indices, dtype=np.int64),
        "frame_index": np.array(all_frame_indices, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(all_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(all_progress_dense, dtype=np.float32)

    df = pa.table(table_data).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)
    final_table = pa.Table.from_pandas(df, preserve_index=False)

    metadata = {b"reward_model_path": reward_model_path.encode()}
    final_table = final_table.replace_schema_metadata(metadata)

    output_path = Path(dataset.root) / "sarm_progress.parquet" if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")

    if "progress_sparse" in df.columns:
        valid = df["progress_sparse"].dropna()
        logging.info(
            f"Sparse progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )
    if "progress_dense" in df.columns:
        valid = df["progress_dense"].dropna()
        logging.info(
            f"Dense progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    if num_visualizations > 0:
        viz_episodes = list(range(min(num_visualizations, num_episodes)))
        logging.info(f"Generating {len(viz_episodes)} visualizations...")
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
            batch_size=batch_size,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute SARM progress values for RA-BC weighting or visualize SARM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--reward-model-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--head-mode", type=str, default="sparse", choices=["sparse", "dense", "both"]
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./sarm_viz")
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames per GPU forward pass (default: 32)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        temp_dataset = LeRobotDataset(args.dataset_repo_id, download_videos=False)
        parquet_path = Path(temp_dataset.root) / "sarm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)"
            )

    if args.visualize_only:
        dataset, reward_model, preprocess = load_sarm_resources(
            args.dataset_repo_id, reward_model_path, args.device
        )
        logging.info(f"Visualization-only mode: visualizing {args.num_visualizations} episodes")
        viz_episodes = list(range(min(args.num_visualizations, dataset.num_episodes)))
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=args.head_mode,
            output_dir=Path(args.output_dir),
            stride=args.stride,
            batch_size=args.batch_size,
        )
        print(f"\nVisualizations saved to: {Path(args.output_dir).absolute()}")
        return

    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    print(f"\nSARM progress values saved to: {output_path}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        hub_path = "sarm_progress.parquet"
        print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hub_path,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(
            f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
        )
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print("  rabc_head_mode: sparse  # or dense")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
