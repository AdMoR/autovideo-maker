import json
import os
import argparse
from typing import NamedTuple, Any, Callable, List
from collections import defaultdict
import datetime
import random
from textwrap import wrap
from collections import Counter
import shutil

from scripts.ffmpeg_utils import *
from scripts.utils import make_dir
from scripts.base_video_classes import VideoElement, VideoDescriptor
from scripts.story_understanding import find_largest_coref_prompt_in_sentence


N_GENERATION_ROUNDS = 1
N_IMAGE_PER_PROMPT = 1


def parse_script_and_scene(lines, separator=".", to_replace=";!,—:", to_clean="/'", mode=None):
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
            if mode is None:
                sequences.append((name, sub, prompt))
            elif mode == "no_prompt":
                sequences.append((name, sub, sub))
            else:
                raise Exception("Unknown prompt mode")
    return sequences, title


def parse_script_and_scene_with_char_config(lines, separator=".", to_replace=";!,—:", to_clean="/'", char_config=None):
    """
    1 - Check the format
    2 - On dialogue lines, extract name and speech
    3 - Decompose in smaller chunks based on separators
    """
    sequences = list()
    title = None
    for l in lines:
        # 1 - Check the format
        script = l.strip().replace('"', "")
        # 2 - We are on a dialogue line, extract name and line
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
            char_prompt, rest = find_largest_coref_prompt_in_sentence(sub, char_config)
            if len(sub) < 3:
                continue
            sequences.append((name, sub, char_prompt + ", " + rest))
    return sequences, title


class Dialogue2Video(NamedTuple):
    dialogue_path: str
    output_dir_prefix: str
    audio_lib_folder: str = "/home/amor/Documents/code_dw/ai-pokemon-episode/audio_lib"
    image_prompt_template: str = "{}, anime art, an image from Pokemon the film, high quality"
    title_prompt: str = "sunset, bright colors, no_humans, panorama, great details"
    final_directory: str = "./finished_clips"
    closing_prompt: str = "sunset, bright colors, no_humans, panorama, great details"

    @property
    def output_dir(self):
        script_name = os.path.basename(self.dialogue_path).split(".")[0]
        return f"{self.output_dir_prefix}/{script_name}"

    @property
    def configuration_path(self):
        return f"{self.output_dir}/raw_video_chunks_configuration.json"

    def main(self):
        make_dir(self.output_dir)
        for i in range(N_GENERATION_ROUNDS * N_IMAGE_PER_PROMPT):
            audio_path = self.find_soundtrack()
            with open(f"{self.output_dir}/soundtrack_{i}.txt", "w") as f:
                f.write(audio_path)
            path, all_video_elements = self.dialogue_to_video(audio_path, image_index_default=i)
            if path is None:
                return
            else:
                self.save_configuration(audio_path, all_video_elements)
            suffix = f"{os.path.basename(self.dialogue_path)}_{i}"
            final_path = self.cleanup(path, suffix=suffix)
            print(f"Clip available under {final_path}")

    def dialogue_to_video(self, audio_path, image_index_default=0):
        make_dir(self.output_dir)
        concat_file_path = f"{self.output_dir}/filelist.txt"

        with open(self.dialogue_path) as f:
            lines = f.readlines()

        print(lines)
        if len(lines) < 5:
            print("Exiting early because of poor script")
            return None, None

        seqs, title = parse_script_and_scene(lines)

        video_part_paths = list()
        all_video_elements = list()
        if title:
            serialization_path = f"{self.output_dir}/serialized_title.json"
            ve = self.gen_or_build_ve(None, title, self.title_prompt, serialization_path, -1)
            all_video_elements.append(ve)
            out_path = ve.to_video(image_index_default)
            video_part_paths.append(out_path)

        print("====> ", self.output_dir, seqs, lines)
        for i, (name, text, prompt) in enumerate(seqs):

            # 2 - Gen images, audio and video
            serialization_path = f"{self.output_dir}/serialized_{i}.json"
            ve = self.gen_or_build_ve(name, text, prompt, serialization_path, i)
            all_video_elements.append(ve)

            # 3.c - Add to the set to merge
            video_path = ve.to_video(img_index=image_index_default)
            video_part_paths.append(video_path)

        closing = True
        if closing:
            serialization_path = f"{self.output_dir}/serialized_title.json"
            ve = self.gen_or_build_ve(None, "@TrueBookWisdom", self.title_prompt, serialization_path, -2)
            all_video_elements.append(ve)
            out_path = ve.to_video(image_index_default)
            video_part_paths.append(out_path)

        # 4.a - Combine all the video chunks together
        combined_name = f"{self.output_dir}/output_version={image_index_default}.mp4"
        combine_part_in_concat_file(video_part_paths, concat_file_path, combined_name)
        # 4.b - And add the background soundtrack
        final_path = f"{self.output_dir}/final_version={image_index_default}.mp4"
        add_soundtrack(combined_name, audio_path, final_path)
        return final_path, all_video_elements

    def save_configuration(self, audio_path, all_video_elements):
        vd = VideoDescriptor(self, audio_path, all_video_elements)
        json.dump(vd.serialize(), open(self.configuration_path, "w"))

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
        return dst

    def gen_or_build_ve(self, name, text, prompt, serialization_path, i):
        if not os.path.exists(serialization_path):
            ve = VideoElement.from_txt_args(name, text, prompt, self.output_dir, i)
            ve.save_serialized(serialization_path)
            del ve
        return VideoElement.load_serialized(serialization_path)
