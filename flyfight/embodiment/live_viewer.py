"""Wall-clock GUI, isolated from the authoritative simulation state.

The single-slot mailbox bounds memory when training outruns the viewer. Pose
interpolation smooths slow computation; it never advances or feeds back physics.
"""
from collections import deque
from dataclasses import dataclass
import copy
import threading
import time

import mujoco
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    episode: int
    simulation_time: float
    qpos: np.ndarray
    produced_at: float
    text: str


class PoseInterpolator:
    """Interpolate hinge/slide joints and free-joint quaternions correctly."""
    def __init__(self, model, initial_qpos):
        self.model = model
        self.origin = initial_qpos.copy()
        self.target = initial_qpos.copy()
        self.velocity = np.zeros(model.nv)
        self.episode = None
        self.started_at = 0.
        self.duration = 0.
        self.previous_production = None
        self.from_time = self.to_time = 0.

    def sample(self, now):
        alpha = min(1., max(0., (now-self.started_at)/self.duration)) if self.duration else 1.
        qpos = self.origin.copy()
        if alpha >= 1:
            qpos[:] = self.target
        else:
            mujoco.mj_integratePos(self.model, qpos, self.velocity, alpha)
        return qpos, self.from_time + alpha*(self.to_time-self.from_time)

    def accept(self, snapshot, now):
        if snapshot.episode != self.episode:
            # Episode resets teleport intentionally; never interpolate across matches.
            self.origin[:] = snapshot.qpos
            self.target[:] = snapshot.qpos
            self.from_time = self.to_time = snapshot.simulation_time
            self.duration = 0.
        else:
            self.origin[:], self.from_time = self.sample(now)
            self.target[:] = snapshot.qpos
            interval = snapshot.produced_at-self.previous_production
            self.duration = float(np.clip(interval, 1/120, 1.))
            self.to_time = snapshot.simulation_time
        mujoco.mj_differentiatePos(self.model, self.velocity, 1., self.origin, self.target)
        self.episode = snapshot.episode
        self.previous_production = snapshot.produced_at
        self.started_at = now


class LiveViewer:
    """Render a separate MjModel/MjData pair at a wall-clock cadence.

    Native viewer mouse forces/sliders affect the display copy only. Neural state,
    physics contacts, damage and RNG never depend on GUI frame rate.
    """
    def __init__(self, env, handle_factory=None):
        if handle_factory is None:
            from mujoco.viewer import launch_passive
            handle_factory = launch_passive
        self.model = copy.copy(env.model)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = env.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self.handle = handle_factory(self.model, self.data)
        self.handle.cam.lookat[:] = [0, 0, .5]
        self.handle.cam.distance = env.config['arena']['size']*.8
        self.handle.cam.azimuth = 90
        self.handle.cam.elevation = -65
        self.fps = env.config['arena'].get('gui_fps', 30)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._settled = threading.Event()
        self._pending = None
        self._final_text = ''
        self._error = None
        self._frames = 0
        self._publications = 0
        self._times = deque(maxlen=4096)
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name='flyfight-display', daemon=True)
        self._thread.start()

    def __call__(self, env, flies, episode, simulation_time):
        if self._error is not None:
            raise RuntimeError('GUI display failed') from self._error
        if self._closed.is_set():
            raise KeyboardInterrupt('GUI closed')
        text = []
        for f in flies:
            text.append(f'Fly {f.id}: hunger={f.internal.hunger:.2f} energy={f.internal.energy:.2f} hemolymph={f.body.hemolymph:.2f}')
            if env.config['motor_learning']['enabled']:
                text.append(f'  Learned motor readout: {f.brain.motor.updates} updates | weight norm={np.linalg.norm(f.brain.motor.weights):.3f}')
            text.extend(f'  {k}: {v.functional_modifier:.2f}' for k,v in f.body.parts.items())
        snapshot = Snapshot(episode, simulation_time, env.data.qpos.copy(), time.perf_counter(), '\n'.join(text))
        with self._lock:
            self._pending = snapshot
            self._final_text = ''
            self._publications += 1
            self._settled.clear()

    def _run(self):
        interpolator = PoseInterpolator(self.model, self.data.qpos)
        latest = None
        previous_snapshot = None
        speed = 0.
        period = 1/self.fps
        deadline = time.perf_counter()
        try:
            while not self._stop.is_set():
                if not self.handle.is_running():
                    break
                now = time.perf_counter()
                with self._lock:
                    snapshot, self._pending = self._pending, None
                    final_text = self._final_text
                if snapshot is not None:
                    if previous_snapshot is not None and previous_snapshot.episode == snapshot.episode:
                        speed = ((snapshot.simulation_time-previous_snapshot.simulation_time)
                                 / max(1e-9,snapshot.produced_at-previous_snapshot.produced_at))
                    interpolator.accept(snapshot, now)
                    latest = previous_snapshot = snapshot
                qpos, displayed_time = interpolator.sample(now)
                with self.handle.lock():
                    self.data.qpos[:] = qpos
                    self.data.time = displayed_time
                    mujoco.mj_forward(self.model, self.data)
                heading = 'Waiting for simulation...' if latest is None else (
                    f'Episode {latest.episode} | simulated {displayed_time:.3f}s\n'
                    f'Compute speed {speed:.3f}x real time | display target {self.fps} FPS\n'
                    'Smoothed live view (display only; simulation may run in slow motion)')
                self.handle.set_texts((None, None, heading+'\n'+(latest.text if latest else '')+'\n'+final_text, ''))
                self.handle.sync(state_only=True)
                with self._lock:
                    self._frames += 1
                    self._times.append(time.perf_counter())
                    if self._pending is None and now >= interpolator.started_at+interpolator.duration:
                        self._settled.set()
                deadline += period
                if deadline < time.perf_counter():
                    deadline = time.perf_counter()
                self._stop.wait(max(0., deadline-time.perf_counter()))
        except Exception as error:
            self._error = error
        finally:
            self._closed.set()
            self._settled.set()

    def finish(self, summary):
        with self._lock:
            self._final_text = f"Winner: {summary['winner']} ({summary['outcome']})"
        self._settled.wait(timeout=1.2)
        if self._error is not None:
            raise RuntimeError('GUI display failed') from self._error

    def metrics(self):
        with self._lock:
            intervals = np.diff(list(self._times))
            return dict(target_fps=self.fps, frames=self._frames, source_updates=self._publications,
                        observed_fps=float(1/np.mean(intervals)) if len(intervals) else 0.,
                        median_interval_ms=float(np.median(intervals)*1000) if len(intervals) else 0.,
                        p95_interval_ms=float(np.percentile(intervals,95)*1000) if len(intervals) else 0.,
                        max_interval_ms=float(np.max(intervals)*1000) if len(intervals) else 0.,
                        display_state_isolated=True, error=str(self._error) if self._error else None)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.handle.close()
        self._thread.join(timeout=1)
