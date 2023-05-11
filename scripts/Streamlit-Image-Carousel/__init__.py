import sys

import streamlit as st
import os
import shutil
import json
import streamlit.components.v1 as components
import sys
import json
from typing import NamedTuple

from scripts.base_video_classes import VideoElement, VideoDescriptor, N_IMAGE_PER_PROMPT
from scripts.ffmpeg_utils import combine_part_in_concat_file, add_soundtrack
from scripts.utils import make_dir


imageCarouselComponent = components.declare_component("image-carousel-component",
                                                      path="Streamlit-Image-Carousel/frontend/public")


class VideoRecombinator(NamedTuple):
    conf: str
    frame_config: str = None
    output_dir: str = "./generation/recombination"

    def main(self):
        # 1 - Preapre config
        make_dir(self.output_dir)
        video_part_paths = list()
        vd = VideoDescriptor.from_serialized(json.load(open(self.conf, "r")))
        audio_path = vd.music_used
        frame_indices = json.load(open(self.frame_config, "r"))

        # 2 - Regenerate video elements with the new frame config
        for ve in vd.all_video_elements:
            frame_index = frame_indices[str(ve.generation_index)]
            vpath = ve.to_video(frame_index, 0)
            video_part_paths.append(vpath)

        # 3.a - Combine all the video chunks together
        combined_name = f"{self.output_dir}/output_version=custom.mp4"
        concat_file_path = f"./temp.txt"
        combine_part_in_concat_file(video_part_paths, concat_file_path, combined_name)
        # 3.b - And add the background soundtrack
        final_path = f"{self.output_dir}/final_version=custom.mp4"
        add_soundtrack(combined_name, audio_path, final_path)

    @st.cache_data()
    def copy_images_to_loc(self):
        vd = VideoDescriptor.from_serialized(json.load(open(self.conf, "r")))
        img_dir = vd.all_video_elements[0].output_dir
        gen_name = os.path.basename(img_dir)
        dst_dir = "/home/amor/Documents/code_dw/ai-pokemon-episode/scripts/" + \
            f"Streamlit-Image-Carousel/frontend/public/images/{gen_name}"
        make_dir(dst_dir)

        for f in os.listdir(img_dir):
            if f.endswith("png"):
                src = os.path.join(img_dir, f)
                dst = f"{dst_dir}/{f}"
                shutil.copy2(src, dst)

        return dst_dir, len(vd.all_video_elements)

    def partial_main(self, frame_config):
        # 1 - Preapre config
        make_dir(self.output_dir)
        video_part_paths = list()
        vd = VideoDescriptor.from_serialized(json.load(open(self.conf, "r")))
        audio_path = vd.music_used
        frame_indices = json.load(open(frame_config, "r"))

        # 2 - Regenerate video elements with the new frame config
        for ve in vd.all_video_elements:
            frame_index = frame_indices[str(ve.generation_index)]
            vpath = ve.to_video(frame_index, 0)
            video_part_paths.append(vpath)

        # 3.a - Combine all the video chunks together
        combined_name = f"{self.output_dir}/output_version=custom.mp4"
        concat_file_path = f"./temp.txt"
        combine_part_in_concat_file(video_part_paths, concat_file_path, combined_name)
        # 3.b - And add the background soundtrack
        final_path = f"{self.output_dir}/final_version=custom.mp4"
        add_soundtrack(combined_name, audio_path, final_path)


def carousel_selector_creation(img_list, frame_index, title):

    filtered_imgs = [img for img in img_list if int(img.split("_")[-2]) == frame_index]

    fimg_paths = list()
    for f in filtered_imgs:
        base_name = os.path.basename(f)
        subfolder = os.path.basename(os.path.dirname(f))
        fimg_paths.append(f"images/{subfolder}/{base_name}")

    if title:
        st.text(title)

    selectedImageUrl = imageCarouselComponent(imageUrls=fimg_paths, height=200)

    if selectedImageUrl is not None:
        img_container_1 = st.image(selectedImageUrl)

        base_name = os.path.basename(selectedImageUrl)
        subfolder = os.path.basename(os.path.dirname(selectedImageUrl))
        option = st.selectbox("image selected", fimg_paths,
                              index=fimg_paths.index(f"images/{subfolder}/{base_name}"))
        return option


def save_config(selected_indexes):
    output = dict()
    for p in selected_indexes:
        if p is None:
            continue
        *args, frame_index, selected_index = p.split("_")
        output[frame_index] = int(selected_index.split(".")[0])
    print("save config -----> ", output)
    json.dump(output, open("./saved_indexes.json", "w"))
    return "./saved_indexes.json"


def main(img_dir: str, vr: VideoRecombinator, N: int):

    imageUrls = [f"{img_dir}/{f}" for f in os.listdir(img_dir)]

    selected = list()
    for i in range(-1, N - 1):
        option = carousel_selector_creation(imageUrls, i, f"Frame {i}")
        selected.append(option)

    st.button("Save", on_click=lambda : vr.partial_main(save_config(selected)))


if __name__ == "__main__":
    print("----> ", sys.argv)
    video_config = sys.argv[1]
    vr = VideoRecombinator(video_config)
    dst_dir, N = vr.copy_images_to_loc()
    main(dst_dir, vr, N)