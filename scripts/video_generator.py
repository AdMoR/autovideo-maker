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
from scripts.tts_utils import tts_solero_auto_speaker
from scripts.ffmpeg_utils import *


generator = torch.manual_seed(33)

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

# Use the Euler scheduler here instead
model_id = "Lykon/DreamShaper" #"Lykon/DreamShaper" #"stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")

DEFAULT_NEG_PROMPT = "mask, worst quality, bad anatomy, bad hands, too many arms, too many fingers"
N_STEPS = 50
GUIDANCE_SCALE = 15
IMG_SIZE = 768
N_IMAGE_PER_PROMPT = 3


def parse_script_and_scene(lines, separator=".", to_replace=";!,—:", to_clean="/'"):
    """
    1 - Check the format
    2 - On dialogue lines, extract name and speech
    3 - Decompose in smaller chunks based on separators
    """
    sequences = list()
    title = None
    for l in lines:
        # 1 - Check the format
        l = l.strip().replace('"', "")
        # 1.a - no | means it is not a normal dialogue line
        if "|" not in l:
            # 1.b - however it could be a title line, retrieve title value
            if "title" in l.lower():
                title = l.split(":")[1]
            continue
        # 2 - We are on a dialogue line, extract name and line
        script, prompt = l.split("|")
        # 2.a - There is the right separator
        if ":" in script:
            tokens = script.split(":")
            name, text = tokens[0], ":".join(tokens[1:])
        # 2.b - Default : the narrator
        else:
            name = "Narrator"
            text = script
        # 3 - Decompose in smaller chunks based on separators
        text = text.strip()
        for c in to_replace:
            text = text.replace(c, separator)
        for c in to_clean:
            text = text.replace(c, "")
        for sub in text.split(separator):
            if len(sub) < 3:
                continue
            for ssub in wrap(sub, 75):
                sequences.append((name, ssub, prompt))
    return sequences, title


class VideoElementGenerators:
    tts_func: Any = tts_solero_auto_speaker
    img_gen_func: Callable[[str], Any] = lambda scene_prompt: pipe(scene_prompt, num_inference_steps=N_STEPS,
                                                                   height=IMG_SIZE,
                                                                   width=IMG_SIZE,
                                                                   #output_type="latent",
                                                                   generator=generator,
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
        for _ in range(3):
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
            raise Exception("Wrong image index")
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

    def serialized(self):
        return self._asdict()

    def save_serialized(self, path):
        json.dump(self._asdict(), open(path, "w"))

    @classmethod
    def load_serialized(cls, path):
        return cls(**json.load(open(path, "r")))


class Dialogue2Video(NamedTuple):
    dialogue_path: str
    output_dir_prefix: str
    audio_lib_folder: str = "/home/amor/Documents/code_dw/ai-pokemon-episode/audio_lib"
    image_prompt_template: str = "{}, anime art, an image from Pokemon the film, high quality"
    title_prompt: str = "A beautiful sunset with bright colors, no_humans, panorama"
    final_directory: str = "./finished_clips"

    @property
    def output_dir(self):
        return f"{self.output_dir_prefix}"

    def main(self):
        for i in range(3 * N_IMAGE_PER_PROMPT):
            audio_path = self.find_soundtrack()
            path = self.dialogue_to_video(audio_path, image_index_default=i)
            suffix = f"{os.path.basename(self.dialogue_path)}_{i}"
            final_path = self.cleanup(path, suffix=suffix)
            print(f"Clip available under {final_path}")

    def find_soundtrack(self):
        tracks = [os.path.join(self.audio_lib_folder, f)
                  for f in os.listdir(self.audio_lib_folder) if f.endswith("mp3")]
        return random.sample(tracks, 1)[0]

    def cleanup(self, src, suffix=None):
        if not os.path.exists(self.final_directory):
            os.mkdir(self.final_directory)
        if suffix is None:
            suffix = str(datetime.datetime.now()).replace(' ', '_')
        dst = f"{self.final_directory}/final_{suffix}.mp4"
        shutil.copyfile(src, dst)
        #shutil.rmtree(self.output_dir)
        return dst

    def dialogue_to_video(self, audio_path, image_index_default=0):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        concat_file_path = f"{self.output_dir}/filelist.txt"

        with open(self.dialogue_path) as f:
            lines = f.readlines()

        print(lines)
        if len(lines) < 5:
            print("Exiting early because of poor script")
            return

        seqs, title = parse_script_and_scene(lines)

        video_part_paths = list()
        if title:
            serialization_path = f"{self.output_dir}/serialized_title.json"
            ve = self.gen_or_build_ve(None, title, self.title_prompt, serialization_path, -1)
            out_path = ve.to_video(image_index_default)
            video_part_paths.append(out_path)

        print("====> ", self.output_dir, seqs, lines)
        for i, (name, text, prompt) in enumerate(seqs):

            # 2 - Gen images, audio and video
            serialization_path = f"{self.output_dir}/serialized_{i}.json"
            ve = self.gen_or_build_ve(name, text, prompt, serialization_path, i)

            # 3.c - Add to the set to merge
            video_path = ve.to_video(img_index=image_index_default)
            video_part_paths.append(video_path)

        # 4.a - Combine all the video chunks together
        combined_name = f"{self.output_dir}/output_version={image_index_default}.mp4"
        combine_part_in_concat_file(video_part_paths, concat_file_path, combined_name)
        # 4.b - And add the background soundtrack
        final_path = f"{self.output_dir}/final_version={image_index_default}.mp4"
        add_soundtrack(combined_name, audio_path, final_path)
        return final_path

    def gen_or_build_ve(self, name, text, prompt, serialization_path, i):
        if not os.path.exists(serialization_path):
            ve = VideoElement.from_txt_args(name, text, prompt, self.output_dir, i)
            ve.save_serialized(serialization_path)
            del ve
        return VideoElement.load_serialized(serialization_path)
