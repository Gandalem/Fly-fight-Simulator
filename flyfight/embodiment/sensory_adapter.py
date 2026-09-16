import numpy as np

FEATURES=['opponent_left','opponent_right','opponent_near','opponent_size','opponent_speed',
          'opponent_motion_left','opponent_motion_right','food_left','food_right','food_odor',
          'contact_left','contact_right','hunger','fatigue','injury','hemolymph_deficit']

class SensoryAdapter:
    """Feature encoding into annotated populations; hypothesis, not retinotopy.

    Sequential nonoverlapping shards make choices auditable in mapping.json.
    """
    def __init__(self,graph,config):
        self.graph=graph; self.config=config
        meta=graph.metadata
        cls=meta['class'].fillna('')
        pools={
            'visual':np.flatnonzero((cls=='visual').to_numpy()),
            'olfactory':np.flatnonzero((cls=='olfactory').to_numpy()),
            'contact':np.flatnonzero(cls.str.contains('mechanosensory').to_numpy()),
            'internal':np.flatnonzero(cls.isin(['MBON','DAN']).to_numpy())}
        self.mapping={}
        used={k:0 for k in pools}
        for idx,feature in enumerate(FEATURES):
            key='visual' if idx<7 else 'olfactory' if idx<10 else 'contact' if idx<12 else 'internal'
            pool=pools[key]; n=min(config['adapter']['max_neurons_per_feature'], max(1,len(pool)//8))
            start=used[key]; self.mapping[feature]=pool[start:start+n]; used[key]+=n
        if not any(len(v) for v in self.mapping.values()): raise ValueError('No annotated sensory populations found')

    def features(self,i,positions,headings,velocities,body,internal,contact,food_available):
        others=np.array([j for j in range(len(positions)) if j!=i])
        rel=positions[others,:2]-positions[i,:2]
        k=np.argmin(np.linalg.norm(rel,axis=1)); opponent=others[k]
        delta=rel[k]; d=float(np.linalg.norm(delta))
        angle=np.arctan2(delta[1],delta[0])-headings[i]
        food=-positions[i,:2]; fd=float(np.linalg.norm(food))
        fa=np.arctan2(food[1],food[0])-headings[i]
        motion=velocities[opponent,:2]-velocities[i,:2]
        lateral=-np.sin(headings[i])*motion[0]+np.cos(headings[i])*motion[1]
        visual=body.modifier('Head')
        smell=(body.modifier('AntennaLeft')+body.modifier('AntennaRight'))/2
        tactile=[]
        for side in ['Left','Right']:
            tactile.append(sum(force*body.modifier(part) for part,force in contact.items() if side in part or part in ['Head','Thorax','Abdomen'])/100)
        x=np.array([max(0,np.sin(angle)),max(0,-np.sin(angle)),1/(1+d),min(1,2/max(d,.1)),
                    min(1,np.linalg.norm(motion)/20),max(0,lateral/20),max(0,-lateral/20),
                    max(0,np.sin(fa)),max(0,-np.sin(fa)),np.exp(-fd/5)*food_available,
                    *tactile,internal.hunger,internal.fatigue,body.injury,1-body.hemolymph],np.float32)
        x[:7]*=visual; x[7:10]*=smell
        return np.clip(x,0,1)

    def encode(self,features,body):
        drive=np.zeros(self.graph.n,np.float32)
        for value,feature in zip(features,FEATURES):
            drive[self.mapping[feature]]+=value*self.config['neural']['input_gain']
        drive*=max(.1,body.hemolymph)
        return drive

    def describe(self):
        return {k:self.graph.metadata.neuron_id.iloc[v].tolist() for k,v in self.mapping.items()}
