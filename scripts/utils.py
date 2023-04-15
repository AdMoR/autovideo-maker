import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime
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
model_id = "stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
N_STEPS = 50
GUIDANCE_SCALE = 15


with open("/home/amor/Documents/code_dw/langchain_test/token") as f:
    token = f.readline()

openai.api_key = "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"


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

