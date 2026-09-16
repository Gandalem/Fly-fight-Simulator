from dataclasses import dataclass
from .body import BodyCondition
from .internal_state import InternalState

@dataclass
class Fly:
    id: int
    brain: object
    body: object
    internal: object

    def reset_body(self,config):
        old_energy=self.internal.energy
        self.body=BodyCondition(config)
        self.internal=InternalState(config['body']['initial_energy'] if config['body']['reset_energy'] else old_energy)
