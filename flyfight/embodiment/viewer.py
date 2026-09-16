from PIL import Image,ImageDraw
import imageio.v2 as imageio
from .live_viewer import LiveViewer

class VideoRecorder:
    """Stream frames to ffmpeg; never retain an episode's frames in RAM."""
    def __init__(self,path,fps=25):
        self.writer=imageio.get_writer(str(path),fps=fps,macro_block_size=16)
    def __call__(self,frame,flies,episode,time):
        image=Image.fromarray(frame); draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,72),fill=(12,19,29))
        draw.text((14,10),f'FLYFIGHT | MaleCNS research prototype | Episode {episode} | {time:.2f}s',fill='white')
        for i,f in enumerate(flies):
            x=14+i*(image.width//len(flies))
            draw.text((x,32),f'Fly {f.id}  energy {f.internal.energy:.2f}  hunger {f.internal.hunger:.2f}  hemolymph {f.body.hemolymph:.2f}',fill='white')
            dn=f.brain.graph.populations['descending']
            rate=f.brain.total_spikes[dn].mean()/max(time,.01) if len(dn) else 0
            draw.text((x,49),f'DN {rate:.1f} Hz | Changed synapses: {(f.brain.delta!=0).sum()}',fill=(140,205,250))
            y=90
            for part,state in f.body.parts.items():
                draw.text((x,y),f'{part}: {state.functional_modifier:.2f}',fill=(255,220,180)); y+=13
        self.last_frame=__import__('numpy').asarray(image)
        self.writer.append_data(self.last_frame)
    def finish(self,summary):
        if not hasattr(self,'last_frame'): return
        image=Image.fromarray(self.last_frame); draw=ImageDraw.Draw(image)
        draw.rectangle((200,280,760,330),fill=(12,19,29))
        text=f"Episode {summary['episode_id']} | winner: {summary['winner']} | {summary['outcome']}"
        draw.text((215,300),text,fill='white')
        self.writer.append_data(__import__('numpy').asarray(image))
    def close(self): self.writer.close()
