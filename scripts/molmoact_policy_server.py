#!/usr/bin/env python
"""Non-RTC MolmoAct2 chunked policy server.

This server intentionally matches `lerobot-rollout` for the non-RTC MolmoAct2
case: each request runs `policy.predict_action_chunk()` and returns a full
postprocessed action chunk.  It reuses LeRobot's existing AsyncInference gRPC
proto and sends small custom pickle dictionaries as payloads.
"""

from __future__ import annotations

import argparse
import logging
import pickle  # nosec: trusted local robotics process payloads
import signal
import threading
import time
import traceback
from concurrent import futures
from typing import Any

LOGGER = logging.getLogger("molmoact_policy_server")


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policy-path", default="robot-learning-team43/molmo_b16_lora_reward_10000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-action-mode", default="continuous", choices=("continuous", "discrete"))
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--n-action-steps", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--enable-inference-cuda-graph", type=_str_to_bool, default=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def _load_runtime_imports() -> None:
    global grpc, torch
    global PreTrainedConfig, get_policy_class, make_pre_post_processors
    global make_robot_action, prepare_observation_for_inference
    global services_pb2, services_pb2_grpc, receive_bytes_in_chunks
    global build_dataset_frame, register_third_party_plugins, init_logging

    import grpc as _grpc
    import torch as _torch

    from lerobot.configs import PreTrainedConfig as _PreTrainedConfig
    from lerobot.policies import get_policy_class as _get_policy_class
    from lerobot.policies import make_pre_post_processors as _make_pre_post_processors
    from lerobot.policies.utils import make_robot_action as _make_robot_action
    from lerobot.policies.utils import prepare_observation_for_inference as _prepare_observation_for_inference
    from lerobot.transport import services_pb2 as _services_pb2
    from lerobot.transport import services_pb2_grpc as _services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks as _receive_bytes_in_chunks
    from lerobot.utils.feature_utils import build_dataset_frame as _build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins as _register_third_party_plugins
    from lerobot.utils.utils import init_logging as _init_logging

    grpc = _grpc
    torch = _torch
    PreTrainedConfig = _PreTrainedConfig
    get_policy_class = _get_policy_class
    make_pre_post_processors = _make_pre_post_processors
    make_robot_action = _make_robot_action
    prepare_observation_for_inference = _prepare_observation_for_inference
    services_pb2 = _services_pb2
    services_pb2_grpc = _services_pb2_grpc
    receive_bytes_in_chunks = _receive_bytes_in_chunks
    build_dataset_frame = _build_dataset_frame
    register_third_party_plugins = _register_third_party_plugins
    init_logging = _init_logging




def _receive_stream_bytes(request_iterator, log_prefix: str) -> bytes:
    """Receive a chunked LeRobot transport stream without version-specific kwargs."""
    chunks: list[bytes] = []
    for item in request_iterator:
        chunks.append(item.data)
        if item.transfer_state == services_pb2.TransferState.TRANSFER_END:
            return b"".join(chunks)
    if chunks:
        return b"".join(chunks)
    raise RuntimeError(f"{log_prefix} received an empty byte stream")

def _resolve_action_key_order(policy_action_names: list[str] | None, dataset_action_names: list[str]) -> list[str]:
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        LOGGER.warning(
            "policy.action_feature_names length (%d) != robot action dim (%d); using robot order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(policy_action_names) != set(dataset_action_names):
        LOGGER.warning("policy.action_feature_names keys do not match robot action keys; using robot order")
        return dataset_action_names
    return policy_action_names


class MolmoActPolicyServer:
    """AsyncInference-compatible gRPC servicer for non-RTC MolmoAct2 chunks."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.lock = threading.Lock()
        self.setup: dict[str, Any] | None = None
        self.latest_observation: dict[str, Any] | None = None
        self.preprocessor = None
        self.postprocessor = None
        self.ordered_action_keys: list[str] = []
        self.request_index = 0
        self.shutdown_event = threading.Event()

        self.policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        self.policy_cfg.device = args.device
        self.policy_cfg.inference_action_mode = args.inference_action_mode
        self.policy_cfg.model_dtype = args.model_dtype
        self.policy_cfg.chunk_size = args.chunk_size
        self.policy_cfg.n_action_steps = args.n_action_steps
        self.policy_cfg.num_inference_steps = args.num_inference_steps
        self.policy_cfg.enable_inference_cuda_graph = args.enable_inference_cuda_graph
        self.policy_cfg.rtc_config = None

        policy_class = get_policy_class(self.policy_cfg.type)
        LOGGER.info("Loading MolmoAct2 policy: %s", args.policy_path)
        self.policy = policy_class.from_pretrained(args.policy_path, config=self.policy_cfg)
        self.policy.to(self.device)
        self.policy.eval()
        LOGGER.info(
            "Policy loaded: type=%s device=%s chunk=%d n_action_steps=%d num_inference_steps=%s",
            self.policy_cfg.type,
            self.device,
            self.policy_cfg.chunk_size,
            self.policy_cfg.n_action_steps,
            self.policy_cfg.num_inference_steps,
        )

    def Ready(self, request, context):  # noqa: N802
        LOGGER.info("Client connected: %s", context.peer())
        with self.lock:
            self.latest_observation = None
            self.request_index = 0
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        setup = pickle.loads(request.data)  # nosec: trusted local robotics process
        if not isinstance(setup, dict):
            raise TypeError(f"Expected setup dict, got {type(setup)}")

        action_keys = [str(key) for key in setup["action_feature_names"]]
        self.ordered_action_keys = _resolve_action_key_order(
            getattr(self.policy.config, "action_feature_names", None),
            action_keys,
        )
        self.setup = setup

        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
            "rename_observations_processor": {"rename_map": setup.get("rename_map", {})},
        }
        postprocessor_overrides = {"device_processor": {"device": "cpu"}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.args.policy_path,
            dataset_stats=None,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        LOGGER.info(
            "Client setup: robot=%s fps=%.1f task=%r action_dim=%d",
            setup.get("robot_type"),
            float(setup.get("fps", 30.0)),
            setup.get("task"),
            len(self.ordered_action_keys),
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = _receive_stream_bytes(request_iterator, "[MOLMOACT SERVER] Observation")
        payload = pickle.loads(received_bytes)  # nosec: trusted local robotics process
        if not isinstance(payload, dict):
            raise TypeError(f"Expected observation payload dict, got {type(payload)}")
        with self.lock:
            self.latest_observation = payload
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        if self.setup is None or self.preprocessor is None or self.postprocessor is None:
            return services_pb2.Actions(data=pickle.dumps({"actions": [], "error": "server_not_ready"}))
        with self.lock:
            obs_payload = self.latest_observation
        if obs_payload is None:
            return services_pb2.Actions(data=pickle.dumps({"actions": [], "error": "no_observation"}))

        try:
            response = self._predict_chunk(obs_payload)
        except Exception as exc:  # noqa: BLE001 - return structured failure to robot-side refill thread
            LOGGER.error("MolmoAct chunk prediction failed: %s", exc)
            LOGGER.debug(traceback.format_exc())
            response = {"actions": [], "error": str(exc)}
        return services_pb2.Actions(data=pickle.dumps(response))

    def _predict_chunk(self, obs_payload: dict[str, Any]) -> dict[str, Any]:
        assert self.setup is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None

        start = time.perf_counter()
        raw_observation = obs_payload["observation"]
        obs_frame = build_dataset_frame(self.setup["hw_features"], raw_observation, prefix="observation")
        observation = prepare_observation_for_inference(
            obs_frame,
            self.device,
            self.setup["task"],
            self.setup["robot_type"],
        )
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action_tensor = self.policy.predict_action_chunk(observation)
            processed = self.postprocessor(action_tensor).squeeze(0).cpu()

        actions: list[dict[str, float]] = []
        dataset_features = self.setup["dataset_features"]
        for step in torch.unbind(processed, dim=0):
            action_dict = make_robot_action(step, dataset_features)
            actions.append({key: float(action_dict[key]) for key in self.ordered_action_keys})

        self.request_index += 1
        latency_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "Predicted MolmoAct chunk #%d: len=%d latency=%.1fms obs_ts=%s",
            self.request_index,
            len(actions),
            latency_ms,
            obs_payload.get("timestep"),
        )
        return {
            "actions": actions,
            "request_index": self.request_index,
            "latency_ms": latency_ms,
            "error": None,
        }


def main() -> None:
    args = parse_args()
    _load_runtime_imports()
    register_third_party_plugins()
    init_logging()

    service = MolmoActPolicyServer(args)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ARG001
        LOGGER.info("Received signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    server.start()
    LOGGER.info("MolmoAct policy server listening on %s:%d", args.host, args.port)
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        server.stop(grace=1.0)
        LOGGER.info("MolmoAct policy server stopped")


if __name__ == "__main__":
    main()
