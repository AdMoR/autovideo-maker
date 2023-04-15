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
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import subprocess
import openai
from tts_utils import tts_solero_auto_speaker
from ffmpeg_utils import *


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
model_id = "andite/anything-v4.0" #"stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

DEFAULT_NEG_PROMPT = "simple background, mask, lowres, bad anatomy, bad hands, text, error, missing fingers, " \
                     "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, " \
                     "jpeg artifacts, signature, watermark, username, blurry, huge breasts, large breasts, sexy, " \
                     "sex, nsfw, sexual"
N_STEPS = 150
GUIDANCE_SCALE = 15


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
                                                                   negative_prompt=DEFAULT_NEG_PROMPT,
                                                                   num_images_per_prompt=10,
                                                                   guidance_scale=GUIDANCE_SCALE).images[0]


class VideoElement(NamedTuple):
    prompt: str
    dialogue: str
    speaker: str
    output_dir: str
    images: List[str] = list()
    audios: List[str] = list()
    generation_index: int = 0

    @classmethod
    def from_txt_args(cls, name, text, prompt, index):
        instance = cls(prompt, text, name, generation_index=index)
        instance.gen()
        return instance

    def audio_path(self, index=0):
        return f"{self.output_dir}/output_{self.generation_index}_{index}.wav"

    def image_path(self, index=0):
        return f"{self.output_dir}/output_{self.generation_index}_{index}.png"

    def gen(self):
        """
        For image generation :

        > rez = pipe("prout", num_images_per_prompt=2)
        > rez
        StableDiffusionPipelineOutput(images=[<PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E7D154640>,
            <PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E81C2C400>], nsfw_content_detected=[False, False])
        > rez.images
        [<PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E7D154640>,
         <PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E81C2C400>]
        > rez.images[0]
        <PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E7D154640>
        > rez.images[0].save("test_0.png")

        """
        VideoElementGenerators.tts_func(self.dialogue, self.speaker, self.audio_path())
        self.audios.append(self.audio_path())
        pil_images = VideoElementGenerators.img_gen_func(self.prompt).images
        for i, img in enumerate(pil_images):
            img.save(self.image_path(i))
            self.images.extend(self.image_path(i))

    def to_video(self, output_dir, img_index=0, audio_index=0) -> str:
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
        # 2.a - Combine audio and image
        mp4_file_path = f"{output_dir}/output_{self.generation_index}.mp4"
        combine_img_audio(png_file_path, wav_file_path, mp4_file_path)
        # 2.b - Combine video and subtitle
        final_mp4_file_path = f"{output_dir}/output_{self.generation_index}_subtitled.mp4"
        return add_subtitle(self.dialogue, mp4_file_path, final_mp4_file_path)

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

    tts_func: Any = tts_solero_auto_speaker
    img_gen_func: Callable[[str], Any] = lambda scene_prompt: pipe(scene_prompt, num_inference_steps=N_STEPS,
                                                                   negative_prompt=DEFAULT_NEG_PROMPT,
                                                                   guidance_scale=GUIDANCE_SCALE).images[0]
    title_prompt: str = "A beautiful sunset with bright colors, no_humans, panorama"
    final_directory: str = "./finished_clips"
    iter: int = 0

    @property
    def output_dir(self):
        return f"{self.output_dir_prefix}_{self.iter}"

    def main(self):
        for i in range(10):
            self = self._replace(iter=i)
            audio_path = self.find_soundtrack()
            self.dialogue_to_video(audio_path)
            suffix = f"{os.path.basename(self.dialogue_path)}_{i}"
            final_path = self.cleanup(suffix=suffix)
            print(f"Clip available under {final_path}")

    def find_soundtrack(self):
        tracks = [os.path.join(self.audio_lib_folder, f)
                  for f in os.listdir(self.audio_lib_folder) if f.endswith("mp3")]
        return random.sample(tracks, 1)[0]

    def cleanup(self, suffix=None):
        if not os.path.exists(self.final_directory):
            os.mkdir(self.final_directory)
        src = f"{self.output_dir}/final.mp4"
        if suffix is None:
            suffix = str(datetime.datetime.now()).replace(' ', '_')
        dst = f"{self.final_directory}/final_{suffix}.mp4"
        shutil.copyfile(src, dst)
        #shutil.rmtree(self.output_dir)
        return dst

    def generate_title_vid(self, book_title):
        # 1 - Create the image and audio
        image_path = f"{self.output_dir}/title.jpg"
        prompt = self.image_prompt_template.format(self.title_prompt)
        image = self.img_gen_func(prompt)
        image.save(image_path)

        # 2 - Add the title
        vpath = add_subtitle(book_title, image_path, f"{self.output_dir}/title_vid.mp4", min_duration=2)
        return add_silent_soundtrack(vpath, f"{self.output_dir}/title_vid_audio.mp4")

    def dialogue_to_video(self, audio_path):
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
            out_path = self.generate_title_vid(title)
            video_part_paths.append(out_path)

        print("====> ", self.output_dir, seqs, lines)
        for i, (name, text, prompt) in enumerate(seqs):
            # 1 - Get the audio
            wav_file_path = f"{self.output_dir}/output_{i}.wav"
            # Tts function as it was before : tts_solero_auto_speaker(text, name, wav_file_path)
            # TODO : create an adapter for all TTs interfaces
            # tts.tts_to_file(text=text, speaker=speaker, language=tts.languages[0],  file_path=wav_file_path)
            # is not compatible yet
            self.tts_func(text, name, wav_file_path)

            # 2 - Create the image
            # Previous interface
            # image = pipe(scene_prompt, num_inference_steps=N_STEPS, guidance_scale=GUIDANCE_SCALE).images[0]
            scene_prompt = self.image_prompt_template.format(prompt)
            image = self.img_gen_func(scene_prompt)
            png_file_path = f"{self.output_dir}/output_{i}.png"
            image.save(png_file_path)

            # 3 - Combine sound and image and subtitle
            # 3.a - Combine sound and image
            mp4_file_path = f"{self.output_dir}/output_{i}.mp4"
            combine_img_audio(png_file_path, wav_file_path, mp4_file_path)
            # 3.b - Combine video and subtitle
            final_mp4_file_path = f"{self.output_dir}/output_{i}_subtitled.mp4"
            add_subtitle(text, mp4_file_path, final_mp4_file_path)
            # 3.c - Add to the set to merge
            video_part_paths.append(final_mp4_file_path)

        # 4.a - Combine all the video chunks together
        combine_part_in_concat_file(video_part_paths, concat_file_path,  f"{self.output_dir}/output.mp4")
        # 4.b - And add the background soundtrack
        final_path = f"{self.output_dir}/final.mp4"
        add_soundtrack(f"{self.output_dir}/output.mp4", audio_path, final_path)
