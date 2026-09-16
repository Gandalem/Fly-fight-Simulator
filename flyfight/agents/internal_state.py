from dataclasses import dataclass

@dataclass
class InternalState:
    energy: float=.45
    satiety: float=0.
    fatigue: float=0.

    @property
    def hunger(self): return 1-self.energy

    def step(self,dt,movement,attack,food,hemolymph,config):
        c=config['body']
        spent=dt*(c['basal_cost']+c['movement_cost']*movement+c['attack_cost']*attack)
        self.energy=max(0,min(1,self.energy-spent+food))
        self.satiety=max(0,min(1,self.satiety-dt*.03+food))
        self.fatigue=max(0,min(1,self.fatigue+dt*(c['fatigue_rate']*movement+c['fatigue_rate']*(1-hemolymph)-c['fatigue_recovery'])))
        return spent
