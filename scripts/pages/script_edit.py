# Contents of ~/my_app/pages/page_2.py
import streamlit as st
import requests
import json
import io
import base64
from PIL import Image
import numpy as np


def build_gen(speaker, speech_lines, prompt, image_key, audio_key):

    def image_gen():
        print("----> ", image_key, "   ", prompt)
        url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        r = requests.post(url,
                          data=json.dumps({"prompt": prompt, "steps": 5}),
                          headers={"Content-Type": "application/json"}).json()
        image = None
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            break
        st.session_state[image_key] = image

    def audio_gen():
        sample_rate = 44100  # 44100 samples per second
        seconds = 2  # Note duration of 2 seconds
        frequency_la = 440  # Our played note will be 440 Hz
        # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
        t = np.linspace(0, seconds, seconds * sample_rate, False)
        # Generate a 440 Hz sine wave
        note_la = np.sin(frequency_la * t * 2 * np.pi)

        st.session_state[audio_key] = (note_la, sample_rate)

    def gen():
        audio_gen()
        image_gen()
    return gen


st.markdown("# Load the base script  ️")
script = st.text_area("Script",
                      value="Title: My Video | A landscape \n Narrator: Once upon a time, there was | A landscape")


st.text(script)

inputs_array = list()

for i, line in enumerate(script.split("\n")):
    speech, prompt = line.split("|")
    who, speech = speech.split(":")
    display_image = None
    c1, c2, c3, c4 = st.columns(4, gap="small")

    image_key = f"image_{i}"
    audio_key = f"audio_{i}"
    with c1:
        n_speech = st.text_input("Speech", value=speech)
    with c2:
        n_prompt = st.text_input(f"Prompt {i}", value=prompt)
    with c3:
        st.button(f"Generate {i}", on_click=build_gen(who, speech, n_prompt, image_key, audio_key))
    with c4:
        st.text(i)
        if image_key in st.session_state:
            st.image(st.session_state[image_key])
        if audio_key in st.session_state:
            audio, sample_rate = st.session_state[audio_key]
            st.audio(audio, sample_rate=sample_rate)
    inputs_array.append((who, n_speech, n_prompt))


def save_array():
    str__ = "\n".join(map(lambda x: f"{x[0]}: {x[1]} | {x[2]}", inputs_array))
    st.text(str__)


def gen_all():
    print(inputs_array)
    for i, x in enumerate(inputs_array):
        build_gen(*x, f"image_{i}", f"audio_{i}")()



st.button("Generate all", on_click=gen_all)
st.button("Save", on_click=save_array)




