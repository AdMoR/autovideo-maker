import streamlit as st
import spacy
import pickle
from autovideo.web_api import api
from autovideo.video_generator import parse_script_and_scene
import json
import os
from autovideo.story_understanding import text_reference_resolver, get_random_char_prompt

try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
models = [m["title"] for m in api.get_sd_models()]


@st.cache_data
def get_char_description(name, gender):
    return get_random_char_prompt(gender)[0]


def parse_script():
    example = st.session_state.story_container
    _, clusters = text_reference_resolver(example)
    st.session_state["clusters"] = clusters


def load_clusters():
    return st.session_state["clusters"]


global_pos_prompt = "best quality, 8k"
global_neg_prompt = "bad quality, out of frame, deformed, text"

if "story_container" not in st.session_state:
    example = """
       Pocahontas, a free-spirited and courageous daughter of Chief Powhatan, finds herself drawn to the enigmatic English explorer, Captain John Smith, as he arrives with his crew to establish a new colony.
        Through their encounters in the breathtaking landscapes of towering forests and rolling rivers, Pocahontas and John Smith bridge the gap between their two worlds and embark on a forbidden love that challenges the prejudices of their societies.
        Pocahontas's deep connection to nature and her wise talking animal companions, including the mischievous raccoon Meeko and the wise hummingbird Flit, guide her along her path of self-discovery and understanding.
        As tensions rise between the Native Americans and the settlers, Pocahontas becomes a voice of reason and compassion, striving to prevent violence and foster a spirit of acceptance and respect.
        In a climactic confrontation, Pocahontas risks everything to save John Smith from the brink of destruction, demonstrating the power of love and the strength of unity.
        Though their love is tested and challenged by the clash of cultures, Pocahontas's unwavering spirit and belief in a world where all can coexist inspire both her people and the settlers to find common ground. 
        Pocahontas's remarkable journey ultimately leads to a message of harmony, respect, and the celebration of diversity, leaving a lasting legacy that transcends time and reminds us of the importance of understanding and acceptance. 
       """
else:
    example = st.session_state.story_container
st.markdown("# Character creation️")
script = st.text_area("Script",
                      value=example, key="story_container", on_change=parse_script)

if "clusters" not in st.session_state:
    parse_script()
clusters = load_clusters()

character_config = None
char_config_path = f"character_conf_{hash(script) % 10000}.json"
if os.path.exists(char_config_path):
    character_config = json.load(open(char_config_path))
if "character_config" in st.session_state:
    character_config = st.session_state["character_config"]
    st.text(str(character_config))


def make_gen(char_key, images_key):
    def gen():
        pos = st.session_state[char_key]
        images = api.txt2img(prompt=pos + ", " + global_pos_prompt,
                             negative_prompt=global_neg_prompt, steps=40,
                             width=512, height=512, batch_size=6).images
        st.session_state[images_key] = images
    return gen


def change_model():
    options = {}
    options['sd_model_checkpoint'] = st.session_state["sd_models"]
    api.set_options(options)


st.selectbox("Model", models, key="sd_models", on_change=change_model)

tabs = st.tabs(clusters.keys())


for i, (tab, (name, elements)) in enumerate(zip(tabs, clusters.items())):
    with tab:
        # State keys
        gender_key = f"gender_{i}"
        char_key = f"char_{i}"
        images_key = f"char_images_{i}"

        # UI elements
        st.text(str(name))
        st.text_area("Occurences", value=", ".join(map(str, elements)))
        gender = st.selectbox("Gender", options=["male", "female"], index=0, key=gender_key)

        coref_key = f"coref_clusters_{i + 1}"
        if character_config is not None and coref_key in character_config:
            description = character_config[coref_key]["prompt"]
        else:
            description = get_char_description(char_key, st.session_state[gender_key])
        st.text_area("Character description", key=char_key, value=description)
        c1, c2 = st.columns(2)
        with c1:
            st.button("Alternate description", key=f"alt_{char_key}",
                      on_click=lambda : st.cache_data.clear())
        with c2:
            st.button("Generate image", key=f"gen_{char_key}", on_click=make_gen(char_key, images_key))

        # Image display if computed
        if images_key in st.session_state:
            images = st.session_state[images_key]
            cs = st.columns(len(images))
            for c, img in zip(cs, images):
                with c:
                    st.image(img)


def save_config():
    config = dict()
    for i, (name, elements) in enumerate(clusters.items()):
        char_key = f"char_{i}"
        images_key = f"char_images_{i}"
        if images_key in st.session_state:
            config[name] = {"text_parts": elements, "prompt": st.session_state[char_key]}

    st.session_state["character_config"] = config
    json.dump(config, open("character_conf.json", "w"))


st.button("Save config", on_click=save_config)
