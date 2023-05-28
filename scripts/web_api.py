import urllib
import requests
import json
import subprocess
import base64
from PIL import Image
import io


def txt_to_speech_call(speech_lines, speaker, outpath):
    safe_string = urllib.parse.quote_plus(speech_lines)
    url = f"http://localhost:5002/api/tts?text={safe_string}&speaker_id={speaker}&style_wav=&language_id="

    rez = subprocess.run(["curl", "-L", "-X", "GET", url, "--output", outpath])
    if rez.returncode == 1:
        raise Exception("ffmpeg audio+image failed")

    return outpath


def stable_diff_call(pos_prompt, neg_prompt):
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    r = requests.post(url,
                      data=json.dumps({"prompt": pos_prompt,
                                       "negative_prompt": neg_prompt, "steps": 20,
                                       "hr_scale": 2, "hr_upscaler": "ESRGAN_4x",
                                       "width": 768, "height": 768}),
                      headers={"Content-Type": "application/json"}).json()
    return [Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0]))) for i in r['images']]
