"""Allocation-reduced FlyGym 2.1 hybrid controller; same equations and timestep.

Adapted from FlyGym 2.1 complex_terrain HybridController, Apache-2.0.
Copyright 2023-2026 The NeuroMechFly v2 Authors.
Modifications: cached indices/phase knots, batched spline evaluation and observations.
License: ../../third_party/FlyGym-LICENSE.txt.
"""
import numpy as np
import mujoco
from scipy.interpolate import PPoly
from flygym_demo.complex_terrain import HybridTurningController, HybridControllerObservation, LocomotionAction
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.hybrid_controller import _CORRECTION_VECTORS, _RIGHT_LEG_CORRECTION_SIGN


class CachedHybridTurningController(HybridTurningController):
    def __post_init__(self):
        super().__post_init__()
        indices={dof:i for i,dof in enumerate(self.output_dof_order)}
        self._output_indices=[]
        self._phase_knots=[]
        self._corrections=[]
        self._phase_values=np.array([0.,.8,0.,-.1,0.])
        splines=[self.preprogrammed_steps._psi_funcs[leg] for leg in self.legs]
        if any(not np.array_equal(spline.x,splines[0].x) for spline in splines):
            raise ValueError('Cached controller requires a common step-spline phase grid')
        # Evaluate the original coefficients in one SciPy call. Select each
        # leg's own phase afterward; no interpolation approximation/resampling.
        self._splines=PPoly(np.stack([spline.c for spline in splines],axis=2),splines[0].x,extrapolate='periodic')
        self._neutral=np.stack([self.preprogrammed_steps.neutral_pos[leg][:,0] for leg in self.legs])
        self._diagonal=np.arange(6)
        for leg in self.legs:
            self._output_indices.append(np.array([indices[dof_spec_to_jointdof(leg,spec)] for spec in self.preprogrammed_steps.dofs_per_leg]))
            start,end=self.preprogrammed_steps.swing_period[leg]
            self._phase_knots.append(np.array([start,np.mean([start,end]),end+self.swing_extension,np.mean([end,2*np.pi]),2*np.pi]))
            vector=_CORRECTION_VECTORS[leg[1]]
            self._corrections.append(vector*_RIGHT_LEG_CORRECTION_SIGN if leg.startswith('r') else vector)

    def step(self,descending_signal,obs):
        descending_signal=np.asarray(descending_signal,dtype=float)
        if descending_signal.shape!=(2,): raise ValueError('descending_signal must have shape (2,)')
        self.cpg_network.intrinsic_amps=np.repeat(np.abs(descending_signal[:,np.newaxis]),3,axis=1).ravel()
        freqs=self._base_intrinsic_freqs.copy()
        freqs[:3]*=1 if descending_signal[0]>=0 else -1
        freqs[3:]*=1 if descending_signal[1]>=0 else -1
        self.cpg_network.intrinsic_freqs=freqs
        selected=self._select_retraction_leg(obs)
        if selected is not None and self.retraction_correction[selected]>self.retraction_persistence_initiation_threshold:
            self.retraction_persistence_counter[selected]=1
        self._update_persistence_counter()
        stumbling=self._get_stumbling_mask(obs)
        self.cpg_network.step()
        sampled=self._splines(self.cpg_network.curr_phases)[self._diagonal,self._diagonal]
        all_angles=self._neutral+self.cpg_network.curr_magnitudes[:,None]*(sampled-self._neutral)
        angles=np.empty(len(self.output_dof_order))
        adhesion=np.empty(6,bool)
        net=np.zeros(6)
        for index,leg in enumerate(self.legs):
            self._update_retraction_correction(index,selected)
            self._update_stumbling_correction(index,stumbling[index])
            if self.retraction_correction[index]>0:
                correction=self.retraction_correction[index]
                self.stumbling_correction[index]=0
            else:
                correction=self.stumbling_correction[index]
            phase=self.cpg_network.curr_phases[index]
            leg_angles=all_angles[index]
            correction=min(max(correction,0.),self.max_correction)
            gain=float(np.interp(phase%(2*np.pi),self._phase_knots[index],self._phase_values))
            angles[self._output_indices[index]]=leg_angles+correction*gain*self._corrections[index]
            net[index]=correction*gain
            adhesion[index]=self._get_adhesion_onoff(leg,phase) if self.enable_adhesion else False
        self.last_info=dict(net_corrections=net,retraction_correction=self.retraction_correction.copy(),
                            stumbling_correction=self.stumbling_correction.copy(),stumbling_mask=stumbling.copy(),
                            leg_to_correct_retraction=selected)
        return LocomotionAction(angles,adhesion)


class CachedObservation:
    """Resolve the stock observer's body/geom selection once per compiled world."""
    def __init__(self,sim,fly_name,legs):
        self.sim=sim
        fly=sim.world.fly_lookup[fly_name]
        bodyseg=type(fly).BODY_SEGMENT_CLASS
        order=fly.get_bodysegs_order()
        bodies=sim._internal_bodyids_by_fly[fly_name]
        self.thorax=bodies[order.index(bodyseg('c_thorax'))]
        self.tips=np.array([bodies[order.index(bodyseg(f'{leg}_tarsus5'))] for leg in legs])
        segments=[bodyseg(f'{leg}_{link}') for leg in legs for link in ('tibia','tarsus1','tarsus2')]
        geoms=sim._internal_geomid_by_bodyseg_by_fly[fly_name]
        self.outputs=np.full(sim.mj_model.ngeom,-1,np.int32)
        for i,segment in enumerate(segments): self.outputs[geoms[segment]]=i
        self.ground=np.zeros(sim.mj_model.ngeom,bool)
        self.ground[sim._internal_ground_geom_ids]=True

    def read(self):
        sim=self.sim; data=sim.mj_data
        contacts=data.contact
        g1=contacts.geom1[:data.ncon]; g2=contacts.geom2[:data.ncon]
        i1=self.outputs[g1]; i2=self.outputs[g2]
        active=((i1>=0)&self.ground[g2] | (i2>=0)&self.ground[g1]) & (contacts.exclude[:data.ncon]==0)
        forces=np.zeros((18,3))
        wrench=np.zeros(6)
        for ci in np.flatnonzero(active):
            mujoco.mj_contactForce(sim.mj_model,data,int(ci),wrench)
            force=contacts.frame[ci].reshape(3,3).T @ wrench[:3]
            if i1[ci]>=0: forces[i1[ci]]-=force
            if i2[ci]>=0: forces[i2[ci]]+=force
        return HybridControllerObservation(float(data.xpos[self.thorax,2]),data.xpos[self.tips,2].copy(),
                                           forces.reshape(6,3,3),data.xmat[self.thorax].reshape(3,3)[:,0].copy())
