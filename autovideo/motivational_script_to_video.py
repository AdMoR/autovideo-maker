
import datetime
import sys

from utils import *
from autovideo.video_generator import Dialogue2Video


prompt_ref_2 = "{}, 3D render, 8k, high quality, ultra-detailed"


class Script2Movie(NamedTuple):
    script_path: str
    output_dir: str = "./generation"
    image_prompt_template: str = prompt_ref_2

    def main(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        today = datetime.date.today()
        Dialogue2Video(self.script_path,
                       output_dir_prefix=f"{self.output_dir}/gen_{today}",
                       image_prompt_template=self.image_prompt_template).\
            main()


if __name__ == "__main__":
    file_dir = sys.argv[1]

    paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith("txt")]

    for path in paths:
        print("Working on ", path)
        generator = Script2Movie(path)
        generator.main()
