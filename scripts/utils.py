import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime
from textwrap import wrap

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
model_id = "stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
N_STEPS = 500
GUIDANCE_SCALE = 15


with open("/home/amor/Documents/code_dw/langchain_test/token") as f:
    token = f.readline()

openai.api_key = "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"


def parse_script_and_scene(lines, separator=".", to_replace=";!", to_clean="/"):
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
            name, text = script.split(":")
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
            sequences.append((name, sub, prompt))
    return sequences, title


class TitlesGenerator(NamedTuple):
    base_prompt_path: str = "/home/amor/Documents/code_dw/ai-pokemon-episode/prompts/prompt_title_pokemon_episode.txt"

    def generate(self):
        with open(self.base_prompt_path) as f:
            prompt = f.readlines()

        # create a completion
        prompt = " ".join(prompt)
        completion = openai.Completion.create(model="text-davinci-003", temperature=1.0,
                                              prompt=prompt, max_tokens=500)

        # print the completion
        response = completion.choices[0].text

        return response.strip().split("\n")


class TitleBasedEpisodeGenerator(NamedTuple):
    base_prompt_path: str = "/home/amor/Documents/code_dw/ai-pokemon-episode/prompts/prompt_base_pokemon.txt"

    def generate(self, title):
        with open(self.base_prompt_path) as f:
            template = f.readlines()
        template = " ".join(template)
        full_prompt = template.format(prompt=title)
        completion_2 = openai.Completion.create(model="text-davinci-003",
                                                prompt=full_prompt,
                                                temperature=1.0,
                                                max_tokens=1800)

        # print the completion
        dialogue_response = completion_2.choices[0].text
        return dialogue_response


def combine_part_in_concat_file(video_path_list, concat_file_path, out_path):
    with open(concat_file_path, "w") as f:
        for path in map(lambda x: os.path.basename(x), video_path_list):
            text = f'file {path}\n'
            f.write(text)
    rez = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path,
                          "-c", "copy", out_path])

    if rez.returncode == 1:
        raise Exception("ffmpeg speedup failed")
    return out_path


class Dialogue2Video(NamedTuple):
    dialogue_path: str
    audio_path: str
    output_dir: str
    image_prompt_template: str = "{}, anime art, an image from Pokemon the film, high quality"

    tts_func: Any = tts_solero_auto_speaker
    img_gen_func: Any = lambda scene_prompt: pipe(scene_prompt, num_inference_steps=N_STEPS,
                                                  guidance_scale=GUIDANCE_SCALE).images[0]

    def main(self):
        return self.dialogue_to_video()

    def dialogue_to_video(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        concat_file_path = f"{self.output_dir}/filelist.txt"

        with open(self.dialogue_path) as f:
            lines = f.readlines()

        if len(lines) < 10:
            print("Exiting early because of poor script")
            return

        seqs, title = parse_script_and_scene(lines)

        video_part_paths = list()
        if title:
            out_path = generate_title_vid(title, self.output_dir)
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
        add_soundtrack(f"{self.output_dir}/output.mp4", self.audio_path, final_path)


def generate_title_vid(book_title, output_path,
                       image_prompt="A beautiful sunset with bright colors, digital art, vector art, trending on artstation"):
    # 1 - Create the image and audio
    image_path = f"{output_path}/title.jpg"
    image = pipe(image_prompt, num_inference_steps=N_STEPS, guidance_scale=GUIDANCE_SCALE).images[0]
    image.save(image_path)

    # 2 - Add the title
    vpath =  add_subtitle(book_title, image_path, f"{output_path}/title_vid.mp4", min_duration=1)
    return add_silent_soundtrack(vpath, f"{output_path}/title_vid_audio.mp4")
