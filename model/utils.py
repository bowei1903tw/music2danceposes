from moviepy.editor import *
import os
from natsort import natsorted


class create_video():
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.topk_file = natsorted(os.listdir(os.path.join(config.retrieval_result,'top_k')))
        self.audio = os.listdir(config.user_given_waveform)[0]
    def create(self):
        
        for topk in self.topk_file:
            frame=[]
            file = natsorted(os.listdir(os.path.join(self.config.retrieval_result,'top_k',topk)))
            
            for i,f in enumerate(file):
                if i!=len(file)-1:
                    duration=int(file[i+1].split('.')[0].split('_')[-1])-int(f.split('.')[0].split('_')[-1])
                else:
                    duration=1
                frame.append(ImageClip(os.path.join(self.config.retrieval_result,'top_k',topk,f), transparent=True).set_duration(duration/60)) #
                            
            output =  concatenate_videoclips(frame, method="compose")
            audio_clip = AudioFileClip(os.path.join(self.config.user_given_waveform, self.audio))
            output = output.set_audio(audio_clip).subclip(0,output.duration)    
            output.write_videofile(f"./{self.config.retrieval_result}/{self.config.retrieval_result}_{topk.split('_')[-1]}.mp4",fps=60) 