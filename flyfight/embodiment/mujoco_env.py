import numpy as np
import mujoco
import time
import subprocess
import warnings
from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (make_locomotion_fly, PreprogrammedSteps,
    HybridTurningController, HybridControllerObservation, LocomotionAction, apply_locomotion_action)
from ..combat.body_parts import part_from_segment, LEG_PARTS
from ..combat.collision import Contact
from ..combat.damage import apply_impact

class MuJoCoArena:
    """Real NeuroMechFly morphology, native fly-fly collisions, joint CPG primitives.

    Body morphology is the NMF female template, used as a proxy for MaleCNS.
    No direct neuron-to-muscle correspondence is claimed.
    """
    def __init__(self,config,agents=2):
        if not 2<=agents<=20: raise ValueError('Arena supports 2..20 flies')
        self.config=config; self.agents=agents
        a=config['arena']
        world=FlatGroundWorld(half_size=a['size'])
        self.flies=[]; self.controllers=[]; self.orders=[]
        steps=PreprogrammedSteps()
        for i in range(agents):
            fly=make_locomotion_fly(name=f'fly{i}',colorize=True)
            angle=2*np.pi*i/agents
            pos=[a['starting_distance']/2*np.cos(angle),a['starting_distance']/2*np.sin(angle),a['spawn_height']]
            yaw=angle+np.pi
            world.add_fly(fly,pos,Rotation3D('quat',(np.cos(yaw/2),0,0,np.sin(yaw/2))),
                          bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                          add_ground_contact_sensors=False)
            self.flies.append(fly)
            self.orders.append(fly.get_actuated_jointdofs_order('position'))
        spec=world.mjcf_root
        for geom in spec.geoms:
            for i in range(agents):
                if geom.name.startswith(f'fly{i}/'):
                    geom.contype=1<<i
                    geom.conaffinity=(((1<<agents)-1) ^ (1<<i)) | (1<<21)
        spec.worldbody.add_geom(name='food_marker',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[a['food_radius'],.025,0],pos=[0,0,.025],rgba=[.94,.63,.12,1],contype=0,conaffinity=0)
        spec.worldbody.add_light(pos=[0,0,15],dir=[0,0,-1])
        # Visible arena walls collide with all flies, with no self-collision.
        for axis in range(2):
            for sign in [-1,1]:
                pos=[0,0,1]; pos[axis]=sign*a['size']/2
                size=[a['size']/2,.15,1] if axis==1 else [.15,a['size']/2,1]
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,
                    rgba=[.15,.18,.22,1],contype=1<<21,conaffinity=(1<<agents)-1)
        self.sim=Simulation(world,timestep=a['physics_dt'])
        self.model,self.data=self.sim.mj_model,self.sim.mj_data
        self.thorax_ids=[self.model.body(f'fly{i}/c_thorax').id for i in range(agents)]
        self.geom_owner={}; self.geom_part={}
        for geom in range(self.model.ngeom):
            body=self.model.geom_bodyid[geom]
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
            for i in range(agents):
                if name.startswith(f'fly{i}/'):
                    geomname=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom) or name
                    self.geom_owner[geom]=i; self.geom_part[geom]=part_from_segment(geomname)
                    self.model.geom_contype[geom]=1<<i
                    self.model.geom_conaffinity[geom]=(((1<<agents)-1) ^ (1<<i)) | (1<<21)
                    break
        self.base_gain=self.model.actuator_gainprm.copy()
        self.base_bias=self.model.actuator_biasprm.copy()
        self.base_force=self.model.actuator_forcerange.copy()
        self.actuator_part={}
        for k in range(self.model.nu):
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,k) or ''
            for i in range(agents):
                if name.startswith(f'fly{i}/'):
                    j=int(self.model.actuator_trnid[k,0])
                    if self.model.actuator_trntype[k]==mujoco.mjtTrn.mjTRN_JOINT:
                        bodyname=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.jnt_bodyid[j]))
                        part=part_from_segment(bodyname)
                    else:
                        part=next((p for leg,p in LEG_PARTS.items() if leg in name),'Thorax')
                    self.actuator_part[k]=(i,part)
        for i in range(agents):
            self.controllers.append(HybridTurningController(timestep=a['physics_dt'],
                preprogrammed_steps=steps,output_dof_order=self.orders[i]))
        self.steps=steps
        self.front_indices=[]
        self.joint_limits=[]
        for i,order in enumerate(self.orders):
            index=[]
            for leg in ('lf','rf'):
                for child,axis in ((f'{leg}_coxa','pitch'),(f'{leg}_coxa','roll'),(f'{leg}_tibia','pitch')):
                    index.append(next(k for k,dof in enumerate(order) if dof.child.name==child and dof.axis.value==axis))
            self.front_indices.append(np.array(index).reshape(2,3))
            limits=[]
            for dof in order:
                joint=self.model.joint(f'fly{i}/{dof.parent.name}-{dof.child.name}-{dof.axis.value}')
                limits.append(joint.range.copy() if joint.limited[0] else [-np.inf,np.inf])
            self.joint_limits.append(np.asarray(limits))
        self.front_offsets=np.zeros((agents,2,3))
        self.front_adhesion=np.ones((agents,2))
        self.renderer=None
        self.render_disabled=False; self.last_gpu_check=0.
        self.reset(config['seed'])

    def reset(self,seed):
        self.sim.reset()
        self.model.actuator_gainprm[:]=self.base_gain
        self.model.actuator_biasprm[:]=self.base_bias
        self.model.actuator_forcerange[:]=self.base_force
        self.front_offsets.fill(0)
        self.front_adhesion.fill(1)
        for i,controller in enumerate(self.controllers):
            controller.reset(seed=seed+i)
            action=LocomotionAction(self.steps.default_pose_by_dof_order(self.orders[i]),np.ones(6,bool))
            apply_locomotion_action(self.sim,f'fly{i}',action)
        self.sim.warmup(.02)
        self.last_positions=self.positions().copy()

    def positions(self): return self.data.xpos[self.thorax_ids].copy()

    def headings(self):
        matrices=self.data.xmat[self.thorax_ids].reshape(-1,3,3)
        return np.arctan2(matrices[:,1,0],matrices[:,0,0])

    def step(self,commands,bodies,internals):
        a=self.config['arena']; dt=a['physics_dt']
        totals=np.zeros(self.agents); dealt=np.zeros(self.agents)
        contacts=[]; max_force=np.zeros(self.agents)
        part_force=[{} for _ in bodies]
        for k,(i,part) in self.actuator_part.items():
            mod=bodies[i].modifier(part)*bodies[i].performance()*(1-.7*internals[i].fatigue)
            self.model.actuator_gainprm[k]=self.base_gain[k]*mod
            self.model.actuator_biasprm[k]=self.base_bias[k]*mod
            self.model.actuator_forcerange[k]=self.base_force[k]*mod
        for _ in range(round(a['control_dt']/dt)):
            for i,controller in enumerate(self.controllers):
                obs=HybridControllerObservation.from_sim(self.sim,f'fly{i}')
                signal=np.asarray(commands[i]['legs'])
                action=controller.step(signal,obs)
                action=self.apply_front_motor(i,action,commands[i],dt)
                apply_locomotion_action(self.sim,f'fly{i}',action)
            self.sim.step()
            for ci in range(self.data.ncon):
                con=self.data.contact[ci]
                g1,g2=int(con.geom1),int(con.geom2)
                i,j=self.geom_owner.get(g1),self.geom_owner.get(g2)
                if i is None or j is None or i==j: continue
                f=np.zeros(6); mujoco.mj_contactForce(self.model,self.data,ci,f)
                force=max(0,float(f[0])); p1,p2=self.geom_part[g1],self.geom_part[g2]
                da=apply_impact(bodies[i],p1,force,dt,self.config)
                db=apply_impact(bodies[j],p2,force,dt,self.config)
                totals[i]+=da; totals[j]+=db; dealt[i]+=db; dealt[j]+=da
                max_force[i]=max(max_force[i],force); max_force[j]=max(max_force[j],force)
                part_force[i][p1]=max(part_force[i].get(p1,0),force)
                part_force[j][p2]=max(part_force[j].get(p2,0),force)
                # Keep one contact summary per pair/part in each control interval.
                if len(contacts)<64: contacts.append(Contact(i,j,p1,p2,force,force*dt,tuple(con.pos)))
        pos=self.positions(); distance=np.linalg.norm(pos[:,:2]-self.last_positions[:,:2],axis=1)
        self.last_positions=pos.copy()
        if not np.isfinite(self.data.qpos).all(): raise FloatingPointError('Nonfinite MuJoCo state')
        return dict(contacts=contacts,contact_active=max_force>0,damage_received=totals,damage_dealt=dealt,force=max_force,
                    part_force=part_force,distance=distance)

    def apply_front_motor(self,i,action,command,dt):
        """Independent bounded joint targets, without a timed gesture or strategy."""
        if 'front_joints' not in command: return action
        c=self.config['motor_learning']
        alpha=1-np.exp(-dt/c['actuator_smoothing_tau'])
        target=np.clip(np.asarray(command['front_joints']),-1,1)*np.asarray(c['joint_offset_radians'])
        self.front_offsets[i]+=alpha*(target-self.front_offsets[i])
        self.front_adhesion[i]+=alpha*(np.clip(command['front_adhesion'],0,1)-self.front_adhesion[i])
        angles=action.joint_angles.copy()
        angles[self.front_indices[i]]+=self.front_offsets[i]
        np.clip(angles,self.joint_limits[i][:,0],self.joint_limits[i][:,1],out=angles)
        adhesion=np.array(action.adhesion_onoff,dtype=float,copy=True) if action.adhesion_onoff is not None else np.ones(6)
        # FlyGym leg order is LF, LM, LH, RF, RM, RH. Preserve CPG swing release.
        adhesion[[0,3]]*=self.front_adhesion[i]
        return LocomotionAction(angles,adhesion)

    def frame(self,width=960,height=640):
        if self.render_disabled: return None
        if time.monotonic()-self.last_gpu_check>5:
            self.last_gpu_check=time.monotonic()
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                if result.returncode==0 and float(result.stdout.splitlines()[0])>self.config['runtime']['vram_limit_gib']*1024-256:
                    self.close(); self.renderer=None; self.render_disabled=True
                    warnings.warn('VRAM budget approached; disabling rendering, CPU simulation continues')
                    return None
            except (OSError,subprocess.TimeoutExpired): pass
        try:
            if self.renderer is None: self.renderer=mujoco.Renderer(self.model,height=height,width=width)
        except (MemoryError,RuntimeError) as error:
            if isinstance(error,RuntimeError) and not any(s in str(error).lower() for s in ['memory','alloc']): raise
            self.render_disabled=True
            warnings.warn('Renderer allocation failed; CPU simulation continues')
            return None
        camera=mujoco.MjvCamera(); camera.lookat[:]=[0,0,.5]
        camera.distance=self.config['arena']['size']*.8; camera.azimuth=90; camera.elevation=-65
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None: self.renderer.close()
