import torch
import TTS

language = 'en'
model_id = 'v3_en'
sample_rate = 48000
speaker = 'en_1'
device = torch.device('cuda')
config = {
    "male": [1, 2, 7, 15, 17, 19, 20],
    "female": [3, 6, 8, 10, 11, 12, 14, 16, 18],
     "male_bof": [9, 13],
     "female_bof": [4, 5]
}

speaker_config = {
    "Ash": 'en_18',
    'Misty':  'en_3',
    'Pikachu': 'en_6',
    'Brock': 'en_2',
    "Narrator": 'en_17',

    'Jesse': 'en_8',
    "Voice": 'en_3',
    "James": 'en_3',
    "Team Rocket": 'en_3',
    'Jessie': 'en_8',
    "DEFAULT": "en_7"}




model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_tts',
                                             language=language,
                                             speaker=model_id)


def tts_solero_auto_speaker(text, speaker_name, audio_path):
    speaker_name = speaker_name if speaker_name in speaker_config else "DEFAULT"
    model.save_wav(text=text,
                   audio_path=audio_path,
                   speaker=speaker_config[speaker_name],
                   sample_rate=sample_rate)

from TTS.api import TTS


class TTSTTS():
    model_name = 'tts_models/en/vctk/vits'

    def __init__(self):
        self.tts = TTS(self.model_name, gpu=True)
        self.speakers = self.tts.speakers

    def generate(self, text, speaker_name, out_path):
        speaker = self.speakers[61]
        text = text.replace("/", "")
        self.tts.tts_to_file(text=text, speaker=speaker,
                            file_path=out_path)