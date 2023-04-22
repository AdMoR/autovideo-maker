import sys
import json
from typing import NamedTuple

from scripts.base_video_classes import VideoElement, VideoDescriptor, N_IMAGE_PER_PROMPT
from scripts.ffmpeg_utils import combine_part_in_concat_file, add_soundtrack
from scripts.utils import make_dir


class VideoRecombinator(NamedTuple):
    conf: str
    frame_config: str
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


if __name__ == "__main__":
    conf_file = sys.argv[1]
    frame_config = sys.argv[2]
    VideoRecombinator(conf_file, frame_config).main()
