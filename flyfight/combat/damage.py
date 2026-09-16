def apply_impact(body,part,force,dt,config):
    """Integrate excess normal force over PHYSICS dt, not rendering dt."""
    p=body.parts[part]
    damage=max(0.,force-p.damage_threshold)*dt*config['body']['damage_multiplier']
    actual=min(p.current_integrity,damage)
    p.current_integrity-=actual
    return actual
