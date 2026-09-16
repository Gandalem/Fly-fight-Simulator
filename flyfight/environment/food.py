import numpy as np

class Food:
    def __init__(self,config):
        self.config=config['arena']; self.amount=self.config['food_amount']
    def consume(self,positions,internals,dt,rng):
        result=np.zeros(len(positions))
        distance=np.linalg.norm(positions[:,:2],axis=1)
        eligible=np.flatnonzero((distance<=self.config['food_radius']) & np.array([x.energy<1 for x in internals]))
        owner=None
        if self.amount>0 and len(eligible):
            # Distance arbitration; seed-controlled random tie-breaking, no agent-ID bias.
            nearest=eligible[np.isclose(distance[eligible],distance[eligible].min(),rtol=0,atol=1e-6)]
            owner=int(rng.choice(nearest))
            result[owner]=min(self.amount,self.config['food_rate']*dt,1-internals[owner].energy)
            self.amount-=result[owner]
        if self.amount<=0 and self.config['food_respawn']: self.amount=self.config['food_amount']
        return result,owner
