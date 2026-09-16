import contextlib
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

from flyfight.embodiment.live_viewer import LiveViewer, PoseInterpolator, Snapshot


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody></mujoco>''')


def test_interpolation_quaternion_and_episode_reset():
    m=model(); d=mujoco.MjData(m)
    start=d.qpos.copy(); target=start.copy()
    target[0]=2.; target[3:7]=[0,0,0,1]
    smoother=PoseInterpolator(m,start)
    smoother.accept(Snapshot(1,0.,start,10.,''),10.)
    smoother.accept(Snapshot(1,.01,target,11.,''),11.)
    mid,t=smoother.sample(11.5)
    assert mid[0]==1. and t==.005
    np.testing.assert_allclose(np.linalg.norm(mid[3:7]),1.,atol=1e-12)
    np.testing.assert_allclose(np.abs(mid[[3,6]]),np.sqrt(.5),atol=1e-12)
    np.testing.assert_array_equal(smoother.sample(12.)[0],target)
    smoother.accept(Snapshot(2,0.,start,12.,''),12.)
    np.testing.assert_array_equal(smoother.sample(12.)[0],start)


class FakeHandle:
    def __init__(self,m,d):
        self.cam=SimpleNamespace(lookat=np.zeros(3))
        self.running=True
        self.updated=threading.Event()
        self.data=d
    def lock(self): return contextlib.nullcontext()
    def is_running(self): return self.running
    def set_texts(self,texts): pass
    def sync(self,state_only=False): self.updated.set()
    def close(self): self.running=False


def test_gui_updates_without_physics_and_uses_isolated_state():
    m=model(); d=mujoco.MjData(m)
    env=SimpleNamespace(model=m,data=d,config={'arena':{'size':20,'gui_fps':30}})
    before=d.qpos.copy(); view=LiveViewer(env,handle_factory=FakeHandle)
    try:
        assert view.handle.updated.wait(2)
        assert view.model is not m and not np.shares_memory(view.data.qpos,d.qpos)
        view(env,[],1,0.)
        # Camera/GUI manipulation of display state must never reach physics.
        with view.handle.lock(): view.data.qpos[0]=999
        np.testing.assert_array_equal(d.qpos,before)
        count=view.metrics()['frames']
        for _ in range(2):
            view.handle.updated.clear()
            assert view.handle.updated.wait(2)
        assert view.metrics()['frames']>=count+2
    finally:
        view.close()
    assert not view._thread.is_alive() and view.metrics()['error'] is None
