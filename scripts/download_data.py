"""Download the three official MaleCNS v1.0 tables; stream, verify, atomic rename."""
from pathlib import Path
import hashlib
import json
import urllib.request
import time

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = ['body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'body-neurotransmitters-male-cns-v1.0.feather',
         'connectome-weights-male-cns-v1.0-minconf-0.5.feather']

def download(directory='data/raw'):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': 'male-cns:v1.0', 'source': 'https://male-cns.janelia.org/download/', 'files': []}
    for name in FILES:
        target = folder / name
        digest = hashlib.sha256()
        if not target.exists():
            with urllib.request.urlopen(BASE + name, timeout=120) as response, target.with_suffix('.partial').open('wb') as out:
                size = int(response.headers['Content-Length'])
                total, last = 0, time.monotonic()
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
                    total += len(block)
                    if time.monotonic() - last > 5:
                        print(f'{name}: {total / 1e6:.0f}/{size / 1e6:.0f} MB', flush=True)
                        last = time.monotonic()
                if total != size:
                    raise IOError(f'Truncated download: {total} != {size}')
            target.with_suffix('.partial').replace(target)
        with target.open('rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        manifest['files'].append({'name': name, 'url': BASE + name, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest()})
        print(f'Verified {name}: {target.stat().st_size / 1e6:.1f} MB', flush=True)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    download()
