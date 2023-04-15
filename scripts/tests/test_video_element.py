import os
import shutil
import tempfile
from unittest import TestCase, skip
from scripts.video_generator import VideoElement


class TestVideoElement(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = "./tempdir"
        if not os.path.exists(cls.output_dir):
            os.mkdir(cls.output_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.output_dir)

    def setUp(self) -> None:
        self.ve = VideoElement.from_txt_args("Narrator", "Hello", "A person", self.output_dir, 0)

    def test_gen(self):
        self.ve.gen()
        print(self.ve.images)
        self.assertEqual(len(self.ve.images), 10)

    def test_full(self):
        path = self.ve.to_video()

    def test_serialization(self):
        path = f"{self.output_dir}/serialized.json"
        self.ve.save_serialized(path)
        a = VideoElement.load_serialized(path)

        self.assertDictEqual(a._asdict(), self.ve._asdict())


class TestTileVideoElement(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = "./tempdir"
        if not os.path.exists(cls.output_dir):
            os.mkdir(cls.output_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.output_dir)

    def setUp(self) -> None:
        self.ve = VideoElement.from_txt_args(None, "Hello", "A person", self.output_dir, 0)

    def test_gen(self):
        print(self.ve._asdict())
        self.assertEqual(len(self.ve.audios), 1)
        self.assertIsNone(self.ve.audios[0])

    def test_full(self):
        path = self.ve.to_video()