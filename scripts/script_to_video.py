import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime
import sys

from utils import *


prompt_ref_1 = "{}, high quality, digital art, trending on artstation"
prompt_ref_2 = "{}, 3D render, 4k, Octane render"
prompt_ref_3 = "{}, high quality, digital art, trending on artstation, vector art"


class Script2Movie(NamedTuple):
    script_path: str
    output_dir: str = "./generation"
    image_prompt_template: str = prompt_ref_2

    def main(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        today = datetime.date.today()
        Dialogue2Video(self.script_path,
                       "/home/amor/Documents/code_dw/ai-pokemon-episode/test.mp3",
                       f"{self.output_dir}/gen_{today}",
                       image_prompt_template=self.image_prompt_template).main()


if __name__ == "__main__":
    path = sys.argv[1]
    generator = Script2Movie(path)
    generator.main()
