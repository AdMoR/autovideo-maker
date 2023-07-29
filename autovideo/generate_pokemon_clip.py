import os
import argparse
from typing import NamedTuple, Any, Callable
from collections import defaultdict
import datetime

from utils import *


class Title2Movie(NamedTuple):
    titleGenerator: Any
    title2Dialogue: Any
    dialogue2Movie: Callable[[str, str], None]
    output_dir: str = "./generation"
    generate: bool = False

    def main(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if self.generate:
            dialogue_files = self.dialogue_generator()
        else:
            dialogue_files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir)
                              if f.endswith("txt")]

        today = datetime.date.today() + datetime.timedelta(days=1)
        for i, filepath in enumerate(dialogue_files):
            dialogue_to_video(filepath, f"{self.output_dir}/gen_{today}_{i}")

    def dialogue_generator(self):
        # 1 - Generate all the title
        titles = self.titleGenerator.generate()
        print("====> ", "\n".join(titles))

        files = list()
        for t in titles:
            dialogue = self.title2Dialogue.generate(t)
            for c in [" ", ",", ";", "!", ".", "?", "|", "/"]:
                t = t.replace(c, "_")
            filename = f"{self.output_dir}/{t}.txt"
            with open(filename, "w") as g:
                g.write(dialogue)
            files.append(filename)

        return files


def main():
    tg = TitlesGenerator()
    tbeg = TitleBasedEpisodeGenerator()
    t2m = Title2Movie(tg, tbeg, dialogue_to_video, "./generation")
    t2m.main()


if __name__ == "__main__":
    main()