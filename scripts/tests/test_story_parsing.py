from unittest import TestCase

from scripts.story_understanding import find_largest_coref_prompt_in_sentence


class TestParsing(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = "Pocahontas, a free-spirited and courageous daughter of Chief Powhatan, finds herself drawn " \
                   "to the enigmatic English explorer, Captain John Smith, as he arrives with his crew to establish " \
                   "a new colony.\n    Through their encounters in the breathtaking landscapes of towering forests a" \
                   "nd rolling rivers, Pocahontas and John Smith bridge the gap between their two worlds and embark " \
                   "on a forbidden love that challenges the prejudices of their societies.\n    Pocahontas's deep " \
                   "connection to nature and her wise talking animal companions, including the mischievous raccoon " \
                   "Meeko and the wise hummingbird Flit, guide her along her path of self-discovery and " \
                   "understanding.\n    As tensions rise between the Native Americans and the settlers, " \
                   "Pocahontas becomes a voice of reason and compassion, striving to prevent violence and " \
                   "foster a spirit of acceptance and respect.\n    In a climactic confrontation, Pocahontas " \
                   "risks everything to save John Smith from the brink of destruction, demonstrating the power " \
                   "of love and the strength of unity.\n    Though their love is tested and challenged by the " \
                   "clash of cultures, Pocahontas's unwavering spirit and belief in a world where all can coexist " \
                   "inspire both her people and the settlers to find common ground. \n    Pocahontas's remarkable " \
                   "journey ultimately leads to a message of harmony, respect, and the celebration of diversity, " \
                   "leaving a lasting legacy that transcends time and reminds us of the importance of understanding " \
                   "and acceptance."
        cls.config = {"coref_clusters_1":
                          {"text_parts": ["Pocahontas, a free-spirited and courageous daughter of Chief Powhatan, "
                                          "finds", "herself drawn", "Pocahontas's deep", "her wise", "her along",
                                          "her path", "Pocahontas becomes", "Pocahontas risks",
                                          "Pocahontas's unwavering", "her people", "Pocahontas's remarkable"],
                           "prompt": "[Yana Klochkova | Edith Masai] as a 23 year old woman hat, fit, yellow, "
                                     "suit, 1woman"}
                      }

    def test_prompt_from_line_and_config(self):
        sentences = self.text.split("\n")
        blocks = sentences[0].split(",")
        character_prompt, remaining = find_largest_coref_prompt_in_sentence(blocks[0], self.config)
        self.assertIsNotNone(character_prompt)
        self.assertEqual(remaining, "")