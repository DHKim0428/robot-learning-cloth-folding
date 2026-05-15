#!/usr/bin/env python3
"""Standalone lerobot rollout with arrow-key DAgger controls.

Drop-in replacement for `lerobot-rollout`.  Identical behaviour except that
when --strategy.type=dagger the keyboard gains two extra bindings:

  ← left arrow   Cancel current correction (discard frames, return to PAUSED)
  → right arrow  Save current correction immediately

All other flags are unchanged.

Usage:
    python scripts/rollout_arrows.py \\
        --policy.path=... --strategy.type=dagger ... (same flags as before)
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# lerobot-rollout entry-point imports (verbatim from lerobot_rollout.py)
# --------------------------------------------------------------------------
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot, RobotConfig,
    bi_openarm_follower, bi_so_follower, earthrover_mini_plus, hope_jr,
    koch_follower, omx_follower, openarm_follower, reachy2, so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator, TeleoperatorConfig,
    bi_openarm_leader, bi_so_leader, homunculus, koch_leader, omx_leader,
    openarm_leader, openarm_mini, reachy2_teleoperator, so_leader, unitree_g1,
)
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.highlight import HighlightStrategy
from lerobot.rollout.strategies.sentry import SentryStrategy
from lerobot.rollout.strategies.core import (
    RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action,
)
from lerobot.utils.import_utils import register_third_party_plugins, _pynput_available
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

# --------------------------------------------------------------------------
# DAgger internals we build on
# --------------------------------------------------------------------------
from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.rollout.configs import (
    BaseStrategyConfig, DAggerKeyboardConfig, DAggerPedalConfig,
    DAggerStrategyConfig, HighlightStrategyConfig, RolloutStrategyConfig,
    SentryStrategyConfig,
)
from lerobot.rollout.context import RolloutContext
from lerobot.rollout.strategies.dagger import (
    DAggerPhase, DAggerEvents, DAggerStrategy,
    _teleop_supports_feedback, _teleop_smooth_move_to, _follower_smooth_move_to,
    _init_dagger_pedal,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

# X11 VK codes for arrow keys — stable across layouts and pynput versions.
_XK_LEFT  = 65361   # XK_Left
_XK_RIGHT = 65363   # XK_Right

# --------------------------------------------------------------------------
# pynput import (mirrors dagger.py logic)
# --------------------------------------------------------------------------
PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception:
        PYNPUT_AVAILABLE = False


# --------------------------------------------------------------------------
# Arrow-aware keyboard listener
# --------------------------------------------------------------------------

def _init_keyboard_arrows(
    events: DAggerEvents,
    cancel_ev: Event,
    save_ev: Event,
    cfg: DAggerKeyboardConfig,
):
    """Like _init_dagger_keyboard but also handles ← cancel and → save."""
    if not PYNPUT_AVAILABLE or is_headless():
        logger.warning("Headless / pynput unavailable — keyboard disabled")
        return None

    special_keys = {
        "space": keyboard.Key.space,
        "tab":   keyboard.Key.tab,
        "enter": keyboard.Key.enter,
    }

    def _resolve(key):
        if key == keyboard.Key.esc:
            return "esc"
        # Arrow keys: match by VK code first (reliable), then by enum.
        vk = getattr(key, "vk", None)
        if vk is None:
            vk = getattr(getattr(key, "value", None), "vk", None)
        if vk == _XK_LEFT or key == keyboard.Key.left:
            return "left"
        if vk == _XK_RIGHT or key == keyboard.Key.right:
            return "right"
        for name, pynput_key in special_keys.items():
            if key == pynput_key:
                return name
        if hasattr(key, "char") and key.char:
            return key.char
        return None

    key_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction:   "correction",
    }

    def on_press(key):
        try:
            resolved = _resolve(key)
            if resolved is None:
                return
            if resolved == "esc":
                logger.info("Stop recording...")
                events.stop_recording.set()
                return
            if resolved == "left":
                logger.info("← Cancel key pressed")
                cancel_ev.set()
                return
            if resolved == "right":
                logger.info("→ Save key pressed")
                save_ev.set()
                return
            if resolved in key_to_event:
                events.request_transition(key_to_event[resolved])
            if resolved == cfg.upload:
                events.upload_requested.set()
        except Exception as exc:
            logger.debug("Key error: %s", exc)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    logger.info(
        "Keyboard ready  pause='%s'  correction='%s'  upload='%s'  ←=cancel  →=save  ESC=stop",
        cfg.pause_resume, cfg.correction, cfg.upload,
    )
    return listener


# --------------------------------------------------------------------------
# Modified DAgger strategy
# --------------------------------------------------------------------------

class DAggerStrategyArrows(DAggerStrategy):
    """DAggerStrategy with ← cancel and → save built directly into the listener."""

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._cancel_requested = Event()
        self._save_requested = Event()

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        # Stop the plain listener the parent created and replace it with the
        # arrow-aware one (same thread, same events object, just extra keys).
        if self.config.input_device == "keyboard":
            if self._listener is not None:
                self._listener.stop()
            self._listener = _init_keyboard_arrows(
                self._events,
                self._cancel_requested,
                self._save_requested,
                self.config.keyboard,
            )

    # ------------------------------------------------------------------
    # Corrections-only loop  (record_autonomous=false)
    # ------------------------------------------------------------------

    def _run_corrections_only(self, ctx: RolloutContext) -> None:
        engine      = self._engine
        cfg         = ctx.runtime.cfg
        robot       = ctx.hardware.robot_wrapper
        teleop      = ctx.hardware.teleop
        dataset     = ctx.data.dataset
        events      = self._events
        interpolator = self._interpolator
        features    = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride    = max(1, cfg.interpolation_multiplier)
        task_str         = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds      = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        self._cancel_requested.clear()
        self._save_requested.clear()
        engine.resume()

        last_action: dict[str, Any] | None = None
        start_time  = time.perf_counter()
        record_tick = 0
        recorded    = 0
        logger.info("DAgger corrections-only started (target: %d episodes)", self.config.num_episodes)

        with VideoEncodingManager(dataset):
            try:
                while (
                    recorded < self.config.num_episodes
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # --- phase transitions (Tab / Space) ---
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(old_phase, new_phase, engine, interpolator, ctx, last_action)
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            with self._episode_lock:
                                dataset.save_episode()
                            recorded += 1
                            self._needs_push.set()
                            logger.info("Correction %d/%d saved", recorded, self.config.num_episodes)
                            log_say(f"Correction {recorded} saved", play_sounds)

                    # --- on-demand upload (Enter) ---
                    if events.upload_requested.is_set():
                        events.upload_requested.clear()
                        logger.info("Upload requested by user")
                        self._background_push(dataset, cfg)

                    # --- ← cancel: discard correction, return to PAUSED ---
                    if self._cancel_requested.is_set():
                        self._cancel_requested.clear()
                        if events.phase == DAggerPhase.CORRECTING:
                            with self._episode_lock:
                                dataset.clear_episode_buffer()
                            if teleop is not None and _teleop_supports_feedback(teleop):
                                with contextlib.suppress(Exception):
                                    teleop.enable_torque()
                            events.phase = DAggerPhase.PAUSED
                            logger.info("← Cancel: correction discarded, returned to PAUSED")
                            log_say("Correction cancelled", play_sounds)

                    # --- → save: end correction immediately ---
                    if self._save_requested.is_set():
                        self._save_requested.clear()
                        if events.phase == DAggerPhase.CORRECTING:
                            events.request_transition("correction")
                            logger.info("→ Save: triggered CORRECTING→PAUSED")

                    phase = events.phase
                    obs   = robot.get_observation()

                    if phase == DAggerPhase.CORRECTING:
                        obs_processed    = ctx.processors.robot_observation_processor(obs)
                        teleop_action    = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action     = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action)
                        last_action = robot_action
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if record_tick % record_stride == 0:
                            obs_frame    = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            dataset.add_frame({
                                **obs_frame, **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            })
                        record_tick += 1

                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            robot.send_action(last_action)

                    else:  # AUTONOMOUS
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue
                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            "Record loop is running slower (%.1f Hz) than target (%.0f Hz).",
                            1 / dt, cfg.fps,
                        )

            finally:
                logger.info("Corrections-only loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # Continuous loop  (record_autonomous=true)
    # ------------------------------------------------------------------

    def _run_continuous(self, ctx: RolloutContext) -> None:
        engine      = self._engine
        cfg         = ctx.runtime.cfg
        robot       = ctx.hardware.robot_wrapper
        teleop      = ctx.hardware.teleop
        dataset     = ctx.data.dataset
        events      = self._events
        interpolator = self._interpolator
        features    = ctx.data.dataset_features

        control_interval  = interpolator.get_control_interval(cfg.fps)
        record_stride     = max(1, cfg.interpolation_multiplier)
        task_str          = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds       = cfg.play_sounds
        episode_duration_s = self._episode_duration_s

        engine.reset()
        interpolator.reset()
        events.reset()
        self._cancel_requested.clear()
        self._save_requested.clear()
        engine.resume()

        last_action: dict[str, Any] | None = None
        record_tick        = 0
        start_time         = time.perf_counter()
        episode_start      = time.perf_counter()
        episodes_since_push = 0
        logger.info("DAgger continuous recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
                while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(old_phase, new_phase, engine, interpolator, ctx, last_action)
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None

                    # --- ← cancel in continuous: just exit correction (mixed buffer) ---
                    if self._cancel_requested.is_set():
                        self._cancel_requested.clear()
                        if events.phase == DAggerPhase.CORRECTING:
                            events.request_transition("correction")
                            logger.info("← Cancel (continuous): exited correction, frames kept")
                            log_say("Correction cancelled", play_sounds)

                    # --- → save: force episode rotation ---
                    if self._save_requested.is_set():
                        self._save_requested.clear()
                        if events.phase != DAggerPhase.CORRECTING:
                            with self._episode_lock:
                                dataset.save_episode()
                            episodes_since_push += 1
                            self._needs_push.set()
                            episode_start = time.perf_counter()
                            logger.info("→ Save: episode %d saved on demand", dataset.num_episodes)
                            log_say(f"Episode {dataset.num_episodes} saved", play_sounds)
                            if episodes_since_push >= self.config.upload_every_n_episodes:
                                self._background_push(dataset, cfg)
                                episodes_since_push = 0

                    phase = events.phase
                    obs   = robot.get_observation()

                    if phase == DAggerPhase.CORRECTING:
                        obs_processed    = ctx.processors.robot_observation_processor(obs)
                        teleop_action    = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action     = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action)
                        last_action = robot_action
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if record_tick % record_stride == 0:
                            obs_frame    = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            dataset.add_frame({
                                **obs_frame, **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            })
                        record_tick += 1

                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            robot.send_action(last_action)

                    else:  # AUTONOMOUS
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue
                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            if record_tick % record_stride == 0:
                                obs_frame    = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                dataset.add_frame({
                                    **obs_frame, **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([False], dtype=bool),
                                })
                            record_tick += 1

                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s and phase != DAggerPhase.CORRECTING:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info("Episode saved (total: %d, elapsed: %.1fs)", dataset.num_episodes, elapsed)
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)
                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0
                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            "Record loop is running slower (%.1f Hz) than target (%.0f Hz).",
                            1 / dt, cfg.fps,
                        )

            finally:
                logger.info("Continuous loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")


# --------------------------------------------------------------------------
# Strategy factory
# --------------------------------------------------------------------------

def _create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    if config.type == "base":
        return BaseStrategy(config)
    if config.type == "sentry":
        return SentryStrategy(config)
    if config.type == "highlight":
        return HighlightStrategy(config)
    if config.type == "dagger":
        return DAggerStrategyArrows(config)
    raise ValueError(f"Unknown strategy type '{config.type}'")


# --------------------------------------------------------------------------
# Entry point  (verbatim copy of lerobot_rollout.rollout, uses our factory)
# --------------------------------------------------------------------------

@parser.wrap()
def rollout(cfg: RolloutConfig):
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)

    strategy = _create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s  (arrow keys: ← cancel  → save)", cfg.strategy.type)

    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Rollout finished")


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
