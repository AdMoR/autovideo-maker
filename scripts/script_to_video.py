import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime
import sys
import random

from utils import *
from scripts.video_generator import Dialogue2Video


prompt_ref_2 = "{}, 3D render, 8k, high quality, ultra-detailed"


class Script2Movie(NamedTuple):
    script_path: str
    output_dir: str = "./generation"
    image_prompt_template: str = prompt_ref_2

    def main(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        today = datetime.date.today()
        script_name = os.path.basename(self.script_path).split(".")[0]
        base_dir = f"{self.output_dir}/gen_{today}"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        Dialogue2Video(self.script_path,
                       output_dir_prefix=f"{base_dir}/{script_name}",
                       image_prompt_template=self.image_prompt_template).\
            main()


if __name__ == "__main__":
    file_dir = sys.argv[1]

    paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith("txt")]

    for path in paths:
        print("Working on ", path)
        generator = Script2Movie(path)
        generator.main()
