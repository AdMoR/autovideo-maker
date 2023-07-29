import os
import argparse
from typing import NamedTuple, Any, Callable


class TitlesGenerator(NamedTuple):
    base_prompt_path: str = "//prompts/prompt_title_pokemon_episode.txt"

    def generate(self):
        import openai
        with open(self.base_prompt_path) as f:
            prompt = f.readlines()

        # create a completion
        prompt = " ".join(prompt)
        completion = openai.Completion.create(model="text-davinci-003", temperature=1.0,
                                              prompt=prompt, max_tokens=500)

        # print the completion
        response = completion.choices[0].text

        return response.strip().split("\n")


class TitleBasedEpisodeGenerator(NamedTuple):
    base_prompt_path: str = "//prompts/prompt_base_pokemon.txt"

    def generate(self, title):
        import openai
        with open(self.base_prompt_path) as f:
            template = f.readlines()
        template = " ".join(template)
        full_prompt = template.format(prompt=title)
        completion_2 = openai.Completion.create(model="text-davinci-003",
                                                prompt=full_prompt,
                                                temperature=1.0,
                                                max_tokens=1800)

        # print the completion
        dialogue_response = completion_2.choices[0].text
        return dialogue_response


def make_dir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        make_dir(dirname)
    if not os.path.exists(path):
        os.mkdir(path)
