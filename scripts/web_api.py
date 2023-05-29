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


import webuiapi


# create API client with custom host, port
api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

# create API client with custom host, port and https
#api = webuiapi.WebUIApi(host='webui.example.com', port=443, use_https=True)

# create API client with default sampler, steps.
#api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

# optionally set username, password when --api-auth is set on webui.
#api.set_auth('username', 'password')

