def homeostatic_signal(energy_improvement, new_injury, control_dt, config):
    """Measured net energy change/trauma per second; no combat or win bonus.

    Net energy includes movement/basal costs and saturation at full energy.
    This prevents rewarding ingestion which did not actually relieve hunger.
    """
    c=config['learning']
    return max(-5.,min(5.,(c['reward_food']*energy_improvement-c['reward_injury']*new_injury)/control_dt))
