import numpy as np

def populations(metadata):
    sc = metadata['superclass'].fillna('')
    cl = metadata['class'].fillna('')
    nt = metadata['neurotransmitter'].fillna('unknown')
    return {
        'sensory': np.flatnonzero(sc.str.contains('sensory').to_numpy()),
        'descending': np.flatnonzero(sc.str.contains('descending').to_numpy()),
        'motor': np.flatnonzero(sc.str.contains('motor').to_numpy()),
        'DAN': np.flatnonzero(((cl == 'DAN') | (nt == 'dopamine')).to_numpy()),
        'mushroom_body': np.flatnonzero(cl.isin(['Kenyon_Cell', 'MBON', 'DAN']).to_numpy()),
        # Only explicitly annotated candidates; empty is valid, no invented IDs.
        'aggression_candidate': np.flatnonzero(metadata['type'].fillna('').str.match(r'^(aIPg|pC1|pC2)').to_numpy()),
    }
