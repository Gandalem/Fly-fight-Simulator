import sqlite3
import json
from pathlib import Path

class EpisodeLogger:
    def __init__(self,path):
        self.db=sqlite3.connect(Path(path)/'episodes.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS episodes (episode_id INTEGER PRIMARY KEY, summary TEXT NOT NULL)')
    def write(self,episode_id,summary):
        with self.db:
            self.db.execute('INSERT INTO episodes VALUES (?,?)',(episode_id,json.dumps(summary)))
    def close(self): self.db.close()

def read(path):
    with sqlite3.connect(Path(path)/'episodes.sqlite') as db:
        return [json.loads(r[0]) for r in db.execute('SELECT summary FROM episodes ORDER BY episode_id')]
