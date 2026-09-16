from ..combat.body_parts import BodyPart,PARTS,LEG_PARTS

class BodyCondition:
    def __init__(self,config):
        c=config['body']
        self.parts={name:BodyPart(c['max_integrity'],c['max_integrity'],c['damage_force_threshold']) for name in PARTS}
        self.hemolymph=1.
        self.injury=0.
        self.immobile_time=0.

    def modifier(self,name): return self.parts[name].functional_modifier

    def performance(self):
        return self.modifier('Thorax')*self.hemolymph

    def update(self,dt,config,speed):
        c=config['body']
        self.injury=sum(1-p.functional_modifier for p in self.parts.values())/len(self.parts)
        severe=sum(max(0,c['severe_integrity']-p.functional_modifier) for p in self.parts.values())
        self.hemolymph=max(0,self.hemolymph-severe*c['bleeding_rate']*dt)
        self.immobile_time=self.immobile_time+dt if speed<c['immobile_speed'] else 0.
