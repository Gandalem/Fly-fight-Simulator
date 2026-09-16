import json
import platform
import subprocess
import time
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import psutil

class Monitor:
    def __init__(self, config, path=None):
        self.config = config['runtime']
        self.path = Path(path) if path else None
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.start = time.perf_counter()
        self.peak = 0

    def sample(self, steps=0, episodes=0):
        memory=self.process.memory_info()
        rss = memory.rss
        self.peak = max(self.peak,rss)
        elapsed = time.perf_counter()-self.start
        info = dict(ram_gib=rss/2**30, peak_sampled_ram_gib=self.peak/2**30,
                    os_peak_ram_gib=getattr(memory,'peak_wset',self.peak)/2**30,
                    cpu_percent=self.process.cpu_percent(), elapsed=elapsed,
                    steps_per_sec=steps/max(elapsed,1e-9), episodes_per_sec=episodes/max(elapsed,1e-9),
                    neural_vram_mib=0, device='cpu')
        try:
            p = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
            if p.returncode == 0:
                mem, util = map(float,p.stdout.splitlines()[0].split(','))
                info.update(total_gpu_used_mib=mem,gpu_utilization=util)
        except (OSError,subprocess.TimeoutExpired):
            info.update(total_gpu_used_mib=None,gpu_utilization=None)
        if rss > self.config['ram_limit_gib']*2**30:
            raise MemoryError('Configured RAM safety limit reached; checkpoint before increasing population')
        if self.path:
            with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(info)+'\n')
        return info

def provenance(config, graph):
    try:
        r = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=3)
        commit = r.stdout.strip() if r.returncode == 0 else None
    except OSError: commit = None
    import hashlib
    root=Path(__file__).resolve().parent
    source_hash=hashlib.sha256()
    for file in sorted(root.rglob('*.py')):
        source_hash.update(str(file.relative_to(root)).encode()); source_hash.update(file.read_bytes())
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), seed=config['seed'], config=config,
                source_sha256=source_hash.hexdigest(),
                git_commit=commit, dataset=graph.provenance, python=platform.python_version(),
                platform=platform.platform(), cpu=platform.processor(), cpu_count=psutil.cpu_count(),
                total_ram_gib=psutil.virtual_memory().total/2**30,
                library_versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pyarrow','flygym','mujoco','numba']})
