"""Two-pass Arrow batch loader. Never construct dense adjacency or all raw edges."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from numba import njit
from .graph import SharedConnectome

@njit(cache=True)
def _fill(pre, post, weights, cursor, indices, outweights):
    for i in range(len(pre)):
        p = pre[i]
        j = cursor[p]
        indices[j] = post[i]
        outweights[j] = weights[i]
        cursor[p] += 1

def edge_batches(path, ids):
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_file(source)
        required = {'body_pre','body_post','weight'}
        if not required.issubset(reader.schema.names):
            raise ValueError(f'Unexpected MaleCNS schema: {reader.schema}')
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            pre = ids.get_indexer(b.column('body_pre').to_numpy())
            post = ids.get_indexer(b.column('body_post').to_numpy())
            w = b.column('weight').to_numpy()
            mask = (pre >= 0) & (post >= 0) & (w > 0)
            yield pre[mask], post[mask], w[mask], len(w)

def prepare_connectome(raw, output, config):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ann = feather.read_feather(raw/'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    ann = ann.loc[ann.superclass.notna()].copy().sort_values('bodyId').reset_index(drop=True)
    ann = ann.rename(columns={'bodyId':'neuron_id'})
    nt = feather.read_feather(raw/'body-neurotransmitters-male-cns-v1.0.feather', columns=['body','consensus_nt'])
    ann['neurotransmitter'] = ann.neuron_id.map(nt.set_index('body').consensus_nt).fillna('unknown')
    ids = pd.Index(ann.neuron_id)
    counts = np.zeros(len(ann), dtype=np.int64)
    path = raw/'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    raw_count = 0
    for pre, post, w, total in edge_batches(path, ids):
        counts += np.bincount(pre, minlength=len(ann))
        raw_count += total
    indptr = np.r_[0,np.cumsum(counts)]
    total = int(indptr[-1])
    if (total*8 + len(ann)*8) > config['runtime']['ram_limit_gib']*2**30*0.7:
        raise MemoryError('CSR exceeds configured RAM budget')
    indices = np.lib.format.open_memmap(output/'indices.npy', mode='w+', dtype=np.int32, shape=(total,))
    weights = np.lib.format.open_memmap(output/'weights.npy', mode='w+', dtype=np.float32, shape=(total,))
    neural = config['neural']
    signs = ann.neurotransmitter.map(neural['signs']).fillna(neural['signs']['unknown']).to_numpy(np.float32)
    cursor = indptr[:-1].copy()
    for pre, post, w, _ in edge_batches(path, ids):
        values = np.minimum(neural['weight_scale']*np.power(w.astype(np.float32),neural['weight_exponent']), neural['max_base_weight'])*signs[pre]
        _fill(pre, post, values, cursor, indices, weights)
    indices.flush(); weights.flush()
    np.save(output/'indptr.npy', indptr)
    ann.to_parquet(output/'neurons.parquet', index=False)
    manifest = json.loads((raw/'manifest.json').read_text()) if (raw/'manifest.json').exists() else {'dataset':'male-cns:v1.0', 'source':'local files; no download manifest'}
    manifest.update(raw_connections=raw_count, retained_connections=total,
                    excluded_connections=raw_count-total,
                    selection='superclass non-null; induced graph; positive synapse count',
                    neural_weight_config={k:neural[k] for k in ['signs','weight_scale','weight_exponent','max_base_weight']})
    (output/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return load_connectome(output)

def load_connectome(path):
    path = Path(path)
    if not (path/'provenance.json').exists():
        raise FileNotFoundError('Run python -m flyfight prepare-connectome first; synthetic data is never substituted')
    return SharedConnectome(*(np.load(path/f'{name}.npy',mmap_mode='r') for name in ['indptr','indices','weights']),
                            pd.read_parquet(path/'neurons.parquet'),
                            json.loads((path/'provenance.json').read_text()))
