import json
import os
import argparse
from typing import NamedTuple, Any, Callable, List
from collections import defaultdict
import datetime
import random
from textwrap import wrap
import shutil

from TTS.api import TTS
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionLatentUpscalePipeline
import subprocess
import openai
from scripts.tts_utils import TTSTTS, tts_solero_auto_speaker
from scripts.ffmpeg_utils import *
from scripts.utils import make_dir


mapping = defaultdict(lambda : 'female-en-5')
mapping.update({'Jesse': 'female-en-5',
               'Pikachu': 'female-en-5',
               'Misty': 'female-en-5',
               "Ash": 'female-en-5\n',
               "Voice": 'female-pt-4\n',
               "James": 'male-en-2',
               "Narrator": 'male-en-2\n',
               'Brock': 'male-pt-3\n',
               "Team Rocket": "male-en-2",
                'Jessie': 'female-en-5',})

#model_name = TTS.list_models()[0]
#tts = TTS(model_name, gpu=True)


try:
    # Use the Euler scheduler here instead
    model_id = "Lykon/DreamShaper" #"Lykon/DreamShaper" #"stabilityai/stable-diffusion-2-1"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe = pipe.to("cuda")
except:
    pass

DEFAULT_NEG_PROMPT = "mask, worst quality, bad anatomy, bad hands, too many arms, too many fingers, watermark, text, cropped"
N_STEPS = 75
GUIDANCE_SCALE = 15
IMG_SIZE = 768
N_IMAGE_PER_PROMPT = 3
N_GENERATION_ROUNDS = 10

TTS_SINGLETON = TTSTTS()


class VideoElementGenerators:
    tts_func: Any = tts_solero_auto_speaker
    img_gen_func: Callable[[str], Any] = lambda scene_prompt: pipe(scene_prompt, num_inference_steps=N_STEPS,
                                                                   height=IMG_SIZE,
                                                                   width=IMG_SIZE,
                                                                   #output_type="latent",
                                                                   #generator=generator,
                                                                   negative_prompt=DEFAULT_NEG_PROMPT,
                                                                   num_images_per_prompt=N_IMAGE_PER_PROMPT,
                                                                   guidance_scale=GUIDANCE_SCALE)


class VideoElement(NamedTuple):
    prompt: str
    dialogue: str
    speaker: str
    output_dir: str
    images: List[str]
    audios: List[str]
    generation_index: int = 0

    @classmethod
    def from_txt_args(cls, name, text, prompt, output_dir, index):
        instance = cls(prompt, text, name, output_dir, list(), list(), generation_index=index)
        instance.gen()
        return instance

    def audio_path(self, index=0):
        return f"{self.output_dir}/output_{self.generation_index}_{index}.wav"

    def image_path(self, index=0):
        return f"{self.output_dir}/output_{self.generation_index}_{index}.png"

    def gen(self):
        """
        1 - Gen 1 audsio
        2 - Gen low reso images
        3 - Upscale them
        """
        if self.speaker is not None:
            VideoElementGenerators.tts_func(self.dialogue, self.speaker, self.audio_path())
            self.audios.append(self.audio_path())
        else:
            self.audios.append(None)
        pil_images = list()
        for _ in range(N_GENERATION_ROUNDS):
            images = VideoElementGenerators.img_gen_func(self.prompt).images
            pil_images.extend(images)
        for i, img in enumerate(pil_images):
            img.save(self.image_path(i))
            self.images.append(self.image_path(i))

    def to_video(self, img_index=0, audio_index=0) -> str:
        # 1 - Handle inputs
        if len(self.images) == 0 or len(self.audios) == 0:
            self.gen()
        if img_index >= len(self.images):
            raise Exception(f"Wrong image index :  {img_index}")
        if audio_index >= len(self.audios):
            raise Exception("Wrong audio index")
        png_file_path = self.images[img_index]
        wav_file_path = self.audios[audio_index]

        # 2 - Generate the final video
        mp4_file_path = f"{self.output_dir}/output_{self.generation_index}.mp4"
        final_mp4_file_path = f"{self.output_dir}/output_{self.generation_index}_subtitled.mp4"
        if wav_file_path is not None:
            # 2.a - Combine audio and image
            combine_img_audio(png_file_path, wav_file_path, mp4_file_path)
            # 2.b - Combine video and subtitle
            return add_subtitle(self.dialogue, mp4_file_path, final_mp4_file_path)
        else:
            vpath = add_subtitle(self.dialogue, png_file_path, mp4_file_path, min_duration=2)
            return add_silent_soundtrack(vpath, final_mp4_file_path)

    def serialize(self):
        return self._asdict()

    def save_serialized(self, path):
        json.dump(self.serialize(), open(path, "w"))

    @classmethod
    def load_serialized(cls, path):
        return cls(**json.load(open(path, "r")))


class VideoDescriptor(NamedTuple):
    orchestrator: NamedTuple
    music_used: str
    all_video_elements: List[VideoElement]

    def serialize(self):
        d1 = self._asdict()
        d1["orchestrator"] = d1["orchestrator"]._asdict()
        d1["all_video_elements"] = [e.serialize() for e in self.all_video_elements]
        return d1

    @classmethod
    def from_serialized(cls, d):
        ves = [VideoElement(**e) for e in d["all_video_elements"]]
        del d["all_video_elements"]
        return cls(**d, all_video_elements=ves)
