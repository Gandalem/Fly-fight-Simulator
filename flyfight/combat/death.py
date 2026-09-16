from .body_parts import LEG_PARTS

def incapacity(body,config):
    c=config['body']
    if body.modifier('Head')<=c['critical_integrity']: return 'critical_head'
    if body.modifier('Thorax')<=c['critical_integrity']: return 'critical_thorax'
    if body.hemolymph<=c['critical_hemolymph']: return 'hemolymph'
    if sum(body.modifier(p)>0 for p in LEG_PARTS.values())<c['min_legs']: return 'locomotion_loss'
    if body.immobile_time>=c['immobile_timeout']: return 'immobility'
    return None
