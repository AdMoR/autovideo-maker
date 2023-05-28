# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from scripts.ffmpeg_utils import *
from scripts.video_generator import parse_script_and_scene
from scripts.web_api import txt_to_speech_call, stable_diff_call
from shutil import copyfile

output_dir = "./temp"
speakers = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p236', 'p237', 'p238',
            'p239', 'p240', 'p241', 'p243', 'p244', 'p245', 'p246', 'p247', 'p248', 'p249', 'p250', 'p251', 'p252',
            'p253', 'p254', 'p255', 'p256', 'p257', 'p258', 'p259', 'p260', 'p261', 'p262', 'p263', 'p264', 'p265',
            'p266', 'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p276', 'p277', 'p278',
            'p279', 'p280', 'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294',
            'p295', 'p297', 'p298', 'p299', 'p300', 'p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308',
            'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p317', 'p318', 'p323', 'p326',
            'p329', 'p330', 'p333', 'p334', 'p335', 'p336', 'p339', 'p340', 'p341', 'p343', 'p345', 'p347', 'p351',
            'p360', 'p361', 'p362', 'p363', 'p364', 'p374', 'p376']
audio_dir = "/home/amor/Documents/code_dw/ai-pokemon-episode/audio_lib/adventure/"
audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)]


def build_gen(output_dir, speaker, speech_key, prompt_key, image_key, audio_key):

    def image_gen():
        prompt = st.session_state[prompt_key]
        images = stable_diff_call(f"{prompt}, {global_pos_prompt}", global_neg_prompt)
        st.session_state[image_key] = images[0]

    def audio_gen():
        """
        curl -L -X GET 'http://localhost:5002/api/tts?text=kaza+maraviyosa&speaker_id=p243&style_wav=&language_id='
        --output maraviyoza.wav
        """
        speech_lines = st.session_state[speech_key]
        fpath = f"{output_dir}/{audio_key}_{speaker}_{abs(hash(speech_lines) % 10000)}.wav"
        if os.path.exists(fpath):
            st.session_state[audio_key] = fpath
            return

        fpath = txt_to_speech_call(speech_lines, speaker, fpath)
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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    str__ = "\n".join(map(lambda x: f"{x[1]}: {x[2]} | {x[3]}", inputs_array))
    st.text(str__)

    volume_mix = 0.3
    if "volume_mix" in st.session_state:
        volume_mix = st.session_state["volume_mix"]

    for k in st.session_state.keys():

        if "image" in k:
            st.session_state[k].save(f"{output_dir}/{k}.jpg")
        if "audio" in k:
            i = int(k.split("_")[1])
            audio_path = st.session_state[k]
            copyfile(audio_path, f"{output_dir}/audio_{i}.wav")
            #scipy.io.wavfile.write(f"{k}.wav", rate, audio)

    video_chunks = list()
    for i, (speaker, who, n_speech, n_prompt) in enumerate(inputs_array):
        vid_path = to_video(output_dir, n_speech, f"{output_dir}/image_{i}.jpg", f"{output_dir}/audio_{i}.wav",
                            generation_index=i)
        video_chunks.append(vid_path)

    out = combine_part_in_concat_file(video_chunks, f"{output_dir}/temp.txt", f"{output_dir}/video.mp4")
    st.session_state["video"] = add_soundtrack(out, soundtrack_path, f"{output_dir}/final_video.mp4", volume_mix)
    print("---->", st.session_state["video"])


def gen_all():
    elements = list(filter(None, inputs_array))
    print(elements)
    for i, x in enumerate(elements):
        image_key = f"image_{i}"
        audio_key = f"audio_{i}"
        speech_key = f"speech_{i}"
        prompt_key = f"prompt_{i}"
        build_gen(output_dir, x[0], speech_key, prompt_key, image_key, audio_key)()


st.markdown("# Load the base script  ️")
script = st.text_area("Script",
                      value="Title: My Video | A landscape \n Narrator: Once upon a time, there was | A landscape")

inputs_array = list()
sequences, title = parse_script_and_scene(script.split("\n"), mode="no_prompt")

global_pos_prompt = st.text_area("Global positive prompt", "high quality, 8k")
global_neg_prompt = st.text_area("Global negative prompt", "bad quality, low res, bad anatomy, text, nude")


for i, (name, speech, prompt) in enumerate(sequences):

    e = st.empty()

    with e.container():
        c0, c1, c2, c3, c4 = st.columns(5, gap="small")

        image_key = f"image_{i}"
        audio_key = f"audio_{i}"
        speech_key = f"speech_{i}"
        prompt_key = f"prompt_{i}"
        with c0:
            pass
        with c1:
            n_speech = st.text_area(f"Speech {i}", key=speech_key, value=speech)
            speaker = st.selectbox(f"Speaker {i}", speakers, speakers.index("p243"))
        with c2:
            n_prompt = st.text_area(f"Prompt {i}", key=prompt_key, value=prompt)
        with c3:
            st.button(f"Generate {i}", on_click=build_gen(output_dir, speaker, speech_key, prompt_key,
                                                          image_key, audio_key))
        with c4:
            st.text(i)
            if image_key in st.session_state:
                st.image(st.session_state[image_key], width=500)
            if audio_key in st.session_state:
                audio = st.session_state[audio_key]
                st.audio(audio)
        element = (speaker, name, n_speech, n_prompt)
    inputs_array.append(element)

st.button("Generate frames", on_click=gen_all)

soundtrack_path = st.selectbox("Audio track", audio_paths, 0)
smix = st.slider("Soundtrack mix", 0.0, 1.0, 0.3, 0.05, key="volume_mix")

st.button("Save and build video", on_click=save_array)

if "video" in st.session_state:
    st.video(st.session_state["video"])



