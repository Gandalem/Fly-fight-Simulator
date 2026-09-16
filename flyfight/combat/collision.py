from dataclasses import dataclass

@dataclass
class Contact:
    a: int
    b: int
    part_a: str
    part_b: str
    force: float
    impulse: float
    position: tuple
