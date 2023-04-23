import os
from textwrap import wrap
import subprocess
import numpy as np
import imageio
from PIL import Image


def add_subtitle(subtitle, video_path, output_path, min_duration=None) -> str:
    # 2 - Add the title
    subtitle = "\n".join(wrap(subtitle, 25))
    args = ['ffmpeg',
            "-y", '-i',
            video_path,
            '-filter_complex',
            f"drawtext=fontfile=/usr/share/fonts/truetype/Gargi/Gargi.ttf:text='{subtitle}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=7*(h-text_h)/8:box=1:boxcolor=black@0.5:boxborderw=5",
            '-max_muxing_queue_size', '9999',
            '-codec:a', 'copy',
            output_path]
    if min_duration is not None:
        supp_args = ["-loop", "1", "-t", str(min_duration), ]
        for i, t in zip(range(2, 2 + len(supp_args)), supp_args):
            args.insert(i, t)

    print("------> ", args)
    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return output_path


def add_soundtrack(video_path, audio_path, output_path):
    """
    """
    rez = subprocess.run(['ffmpeg', '-y',
                          '-i', video_path, '-i', audio_path,
                          '-filter_complex', '[1:a:0]volume=0.25[a1];[0:a:0][a1]amerge=2[aout]',
                          '-map', '[aout]',
                          '-map', '0:v',
                          "-c:v", "copy",
                          "-c:a", "aac",
                          "-b:a", "192k",
                          "-ar", "44100",
                          output_path])
    if rez.returncode != 0:
        raise Exception(rez)


def add_silent_soundtrack(video_path, out_path):
    args = [
        "ffmpeg", "-y", "-f",
        "lavfi", "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000", "-i",
        video_path, "-shortest", out_path
    ]
    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return out_path


def combine_img_audio(png_file_path, audio_path, mp4_file_path):
    rez = subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-i", png_file_path,
                          "-framerate", "1", mp4_file_path])
    if rez.returncode == 1:
        raise Exception("ffmpeg audio+image failed")
    return mp4_file_path


def combine_part_in_concat_file(video_path_list, concat_file_path, out_path):
    with open(concat_file_path, "w") as f:
        for path in map(lambda x: os.path.basename(x), video_path_list):
            text = f'file {path}\n'
            f.write(text)
    rez = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path,
                          "-c", "copy", out_path])

    if rez.returncode == 1:
        raise Exception("ffmpeg concat failed")
    return out_path


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def add_subtitles(input_vid, subtitles, out_vid):
    """
    Tested with :
    input_vid = 'infinite_zoom_1682173460.123847.mp4'
    out_vid = "out.mp4"
    subtitles = ["ok", "super", "ok"]
    add_subtitles(input_vid, subtitles, out_vid)
    """
    D = get_length(input_vid)

    streams = ["0:v"] + [f"v{i}" for i in range(1, len(subtitles) + 1)]
    N = len(streams) - 1
    elements = list()

    for i, (subt, (in_stream, out_stream)) in enumerate(zip(subtitles,
                                                            zip(streams[:-1], streams[1:]))):
        elements.append(
            f"[{in_stream}]drawtext=fontfile=/usr/share/fonts/truetype/Gargi/Gargi.ttf:text='{subt}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=7*(h-text_h)/8:box=1:boxcolor=black@0.5:boxborderw=5:"
            f"enable='between(t,{round(i / N, 1) * D} ,{round((i + 1) / N, 1) * D})'[{out_stream}]")

    cmd = ";".join(elements)

    args = ['ffmpeg',
            "-y", '-i',
            input_vid,
            '-filter_complex',
            cmd,
            "-map", f"[{streams[-1]}]",
            '-max_muxing_queue_size', '9999',
            '-codec:a', 'copy',
            out_vid]

    rez = subprocess.run(args)
    if rez.returncode == 1:
        raise Exception("ffmpeg multiple subtitles failed")

    return out_vid


def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames = frames[::-1]

    # Get dimensions of the first frames, all subsequent has to be same sized
    for k in frames:
        assert (k.size == frames[0].size,"Different frame sizes found!")

    # Create an imageio video writer, avoid block size of 512.
    writer = imageio.get_writer(file_path, fps=fps, macro_block_size=None)

    # Duplicate the start and end frames
    start_frames = [frames[0]] * start_frame_dupe_amount
    end_frames = [frames[-1]] * last_frame_dupe_amount

    # Write the duplicated frames to the video writer
    for frame in start_frames:
        # Convert PIL image to numpy array
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the frames to the video writer
    for frame in frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the duplicated frames to the video writer
    for frame in  end_frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Close the video writer
    writer.close()


def shrink_and_paste_on_blank(current_image, mask_width, mask_height):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    :param mask_height: height in pixels to shrink from each side
    """

    # calculate new dimensions
    width, height = current_image.size
    new_width = width - 2 * mask_width
    new_height = height - 2 * mask_height

    # resize and paste onto blank image
    prev_image = current_image.resize((new_width, new_height))
    blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 1))
    blank_image.paste(prev_image, (mask_width, mask_height))

    return blank_image
