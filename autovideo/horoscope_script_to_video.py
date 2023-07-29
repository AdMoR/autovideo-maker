import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime
import sys
import random

from utils import *
from autovideo.video_generator import Dialogue2Video


class Script2Movie(NamedTuple):
    script_path: str
    output_dir: str = "./generation"

    def main(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        today = datetime.date.today()
        Dialogue2Video(self.script_path,
                       output_dir_prefix=f"{self.output_dir}/gen_{today}",
                       image_prompt_template="{}, 3D render, 8k, high quality, ultra-detailed",
                       title_prompt="A future seeing crystal ball with fumes in the background, digital art").\
            main()


if __name__ == "__main__":
    file_dir = sys.argv[1]

    paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith("txt")]

    for path in paths:
        print("Working on ", path)
        generator = Script2Movie(path)
        generator.main()
