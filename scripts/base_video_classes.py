import json
import os
import argparse
from typing import NamedTuple, Any, Callable, List
from collections import defaultdict
import datetime
import random
from textwrap import wrap
import shutil
import time

from TTS.api import TTS
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionInpaintPipeline, \
    EulerAncestralDiscreteScheduler
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
        1 - Gen 1 audio
        2 - Gen low res images
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


class ZoomModels:

    @classmethod
    def build_model_fn(cls, negative_prompt, guidance_scale=15, num_inference_steps=50, size=512):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        def no_check(images, **kwargs):
            return images, False

        pipe.safety_checker = no_check
        pipe.enable_attention_slicing()

        return lambda prompt, current_image, mask_image: pipe(prompt=prompt,
             negative_prompt=negative_prompt,
             image=current_image,
             guidance_scale=guidance_scale,
             height=size,
             width=size,
             mask_image=mask_image,
             num_inference_steps=num_inference_steps)


class ZoomVideoElement(NamedTuple):
    prompts_array: List[str]
    dialogue: str
    speaker: str
    output_dir: str
    pipe_fn: Any
    images: List[Any]

    @classmethod
    def from_txt_args(cls, name, text, prompts, output_dir, index):
        instance = cls(prompts, text, name, output_dir, None, list(),)
        instance.gen()
        return instance

    def zoom(self, custom_init_image):
        prompts = {}
        for x in self.prompts_array:
            try:
                key = int(x[0])
                value = str(x[1])
                prompts[key] = value
            except ValueError:
                pass
        num_outpainting_steps = len(self.prompts_array)

        height = 512
        width = height

        current_image = Image.new(mode="RGBA", size=(height, width))
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")
        current_image = current_image.convert("RGB")
        if (custom_init_image):
            current_image = custom_init_image.resize(
                (width, height), resample=Image.LANCZOS)
        else:
            print(">>> ", prompts[min(k for k in prompts.keys() if k >= 0)])
            init_images = self.pipe_fn(prompts[min(k for k in prompts.keys() if k >= 0)],
                                       current_image,
                                       mask_image,)[0]
            current_image = init_images[0]
        mask_width = 128
        num_interpol_frames = 30

        self.images.append(current_image)

        for i in range(num_outpainting_steps):
            print('Outpaint step: ' + str(i + 1) +
                  ' / ' + str(num_outpainting_steps))

            prev_image_fix = current_image
            prev_image = shrink_and_paste_on_blank(current_image, mask_width, mask_width)
            current_image = prev_image

            # create mask (black image with white mask_width width edges)
            mask_image = np.array(current_image)[:, :, 3]
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")

            # inpainting step
            current_image = current_image.convert("RGB")
            print("-----> ", prompts[max(k for k in prompts.keys() if k <= i)])
            images = self.pipe_fn(prompts[max(k for k in prompts.keys() if k <= i)],
                                  current_image,
                                  mask_image, )[0]
            current_image = images[0]
            current_image.paste(prev_image, mask=prev_image)

            # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
            for j in range(num_interpol_frames - 1):
                interpol_image = current_image
                interpol_width = round(
                    (1 - (1 - 2 * mask_width / height) ** (1 - (j + 1) / num_interpol_frames)) * height / 2
                )
                interpol_image = interpol_image.crop((interpol_width,
                                                      interpol_width,
                                                      width - interpol_width,
                                                      height - interpol_width))

                interpol_image = interpol_image.resize((height, width))

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = round(
                    (1 - (height - 2 * mask_width) / (height - 2 * interpol_width)) / 2 * height
                )
                prev_image_fix_crop = shrink_and_paste_on_blank(
                    prev_image_fix, interpol_width2, interpol_width2)
                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

                self.images.append(interpol_image)
            self.images.append(current_image)

        return self.to_video(self.images)

    def to_video(self, all_frames):
        save_path = f"{self.output_dir}/infinite_zoom_{str(time.time())}.mp4"
        fps = 30
        start_frame_dupe_amount = 15
        last_frame_dupe_amount = 15

        write_video(save_path, all_frames, fps, False,
                    start_frame_dupe_amount, last_frame_dupe_amount)
        return save_path
