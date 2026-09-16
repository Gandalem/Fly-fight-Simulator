def homeostatic_signal(food_energy, new_injury, control_dt, config):
    """Improvement/trauma per second, held for one control interval; no win bonus."""
    c=config['learning']
    return max(-5.,min(5.,(c['reward_food']*food_energy-c['reward_injury']*new_injury)/control_dt))
