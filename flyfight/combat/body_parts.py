from dataclasses import dataclass

PARTS=['Head','Thorax','Abdomen','FrontLegLeft','FrontLegRight','MidLegLeft','MidLegRight',
       'HindLegLeft','HindLegRight','WingLeft','WingRight','AntennaLeft','AntennaRight']
LEG_PARTS={'lf':'FrontLegLeft','rf':'FrontLegRight','lm':'MidLegLeft','rm':'MidLegRight','lh':'HindLegLeft','rh':'HindLegRight'}

def part_from_segment(name):
    name=name.split('/')[-1]
    prefix=name.split('_')[0]
    if prefix in LEG_PARTS: return LEG_PARTS[prefix]
    if 'wing' in name: return 'WingLeft' if prefix=='l' else 'WingRight'
    if 'antenna' in name or 'arista' in name: return 'AntennaLeft' if prefix=='l' else 'AntennaRight'
    if any(x in name for x in ['head','eye','rostrum','haustellum','labellum']): return 'Head'
    if 'abdomen' in name: return 'Abdomen'
    return 'Thorax'

@dataclass
class BodyPart:
    max_integrity: float=1.
    current_integrity: float=1.
    damage_threshold: float=15.
    @property
    def functional_modifier(self):
        return max(0.,self.current_integrity/self.max_integrity)
    def severity(self,c):
        x=self.functional_modifier
        return 'lost' if x<=0 else 'severe' if x<c['severe_integrity'] else 'mild' if x<c['mild_integrity'] else 'normal'
