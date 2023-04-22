import os
from textwrap import wrap
import subprocess


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
