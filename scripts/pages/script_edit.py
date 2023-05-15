# Contents of ~/my_app/pages/page_2.py
import streamlit as st
import requests
import json
import io
import base64
from PIL import Image
import numpy as np
import scipy
import os
import urllib.parse
import subprocess
from scripts.ffmpeg_utils import *
from scripts.video_generator import parse_script_and_scene


def build_gen(speaker, speech_lines, prompt, image_key, audio_key):

    def image_gen():
        print("----> ", image_key, "   ", prompt)
        url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        r = requests.post(url,
                          data=json.dumps({"prompt": prompt, "steps": 20}),
                          headers={"Content-Type": "application/json"}).json()
        image = None
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            break
        st.session_state[image_key] = image

    def audio_gen():
        """
        curl -L -X GET 'http://localhost:5002/api/tts?text=kaza+maraviyosa&speaker_id=p243&style_wav=&language_id='
        --output maraviyoza.wav
        """
        fpath = f"{audio_key}.wav"
        if os.path.exists(fpath):
            st.session_state[audio_key] = fpath
            return

        safe_string = urllib.parse.quote_plus(speech_lines)
        url = f"http://localhost:5002/api/tts?text={safe_string}&speaker_id=p243&style_wav=&language_id="

        rez = subprocess.run(["curl", "-L", "-X", "GET", url,  "--output", fpath])
        if rez.returncode == 1:
            raise Exception("ffmpeg audio+image failed")

        st.session_state[audio_key] = fpath

    def gen():
        audio_gen()
        image_gen()
    return gen


def to_video(output_dir, dialogue, png_file_path, wav_file_path, generation_index=0) -> str:
    # 1 - Handle inputs

    # 2 - Generate the final video
    mp4_file_path = f"{output_dir}/output_{generation_index}.mp4"
    final_mp4_file_path = f"{output_dir}/output_{generation_index}_subtitled.mp4"
    if wav_file_path is not None:
        # 2.a - Combine audio and image
        combine_img_audio(png_file_path, wav_file_path, mp4_file_path)
        # 2.b - Combine video and subtitle
        return add_subtitle(dialogue, mp4_file_path, final_mp4_file_path)
    else:
        vpath = add_subtitle(dialogue, png_file_path, mp4_file_path, min_duration=2)
        return add_silent_soundtrack(vpath, final_mp4_file_path)


def save_array():
    output_dir = "./temp"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    str__ = "\n".join(map(lambda x: f"{x[0]}: {x[1]} | {x[2]}", inputs_array))
    st.text(str__)

    for k in st.session_state.keys():

        if "image" in k:
            st.session_state[k].save(f"{output_dir}/{k}.jpg")
        if "audio" in k:
            audio = st.session_state[k]
            #scipy.io.wavfile.write(f"{k}.wav", rate, audio)

    video_chunks = list()
    for i, (who, n_speech, n_prompt) in enumerate(inputs_array):
        vid_path = to_video(output_dir, n_speech, f"{output_dir}/image_{i}.jpg", f"audio_{i}.wav", generation_index=i)
        video_chunks.append(vid_path)

    st.session_state["video"] = combine_part_in_concat_file(video_chunks,
                                                            f"{output_dir}/temp.txt", f"{output_dir}/video.mp4")


def gen_all():
    print(inputs_array)
    for i, x in enumerate(inputs_array):
        build_gen(*x, f"image_{i}", f"audio_{i}")()


st.markdown("# Load the base script  ️")
script = st.text_area("Script",
                      value="Title: My Video | A landscape \n Narrator: Once upon a time, there was | A landscape")


st.text(script)

inputs_array = list()


sequences, title = parse_script_and_scene(script.split("\n"))


for i, (name, speech, prompt) in enumerate(sequences):

    c1, c2, c3, c4 = st.columns(4, gap="small")

    image_key = f"image_{i}"
    audio_key = f"audio_{i}"
    with c1:
        n_speech = st.text_area("Speech", value=speech)
    with c2:
        n_prompt = st.text_area(f"Prompt {i}", value=prompt)
    with c3:
        st.button(f"Generate {i}", on_click=build_gen(name, speech, n_prompt, image_key, audio_key))
    with c4:
        st.text(i)
        if image_key in st.session_state:
            st.image(st.session_state[image_key], width=500)
        if audio_key in st.session_state:
            audio = st.session_state[audio_key]
            st.audio(audio)
    inputs_array.append((name, n_speech, n_prompt))


if "video" in st.session_state:
    st.video(st.session_state["video"])


st.button("Generate all", on_click=gen_all)
st.button("Save", on_click=save_array)




