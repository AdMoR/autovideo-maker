import os
import pandas as pd
import random
import spacy
from collections import defaultdict

# import dataclass


def get_gendered_celebs(gender):
    if gender not in ["male", "female"]:
        raise Exception("Use male or female")
    path = f"/home/amor/Documents/code_dw/ai-pokemon-episode/external_data/{gender}_names.txt"

    with open(path) as f:
        lines = f.readlines()

    names = set()

    for l in lines:
        name = " ".join(os.path.basename(l.strip()).split("_")[:-1])
        if len(name) < 2:
            continue
        names.add(name)

    return names


def build_celeb_df():
    data = list()
    for g in ["male", "female"]:
        names = get_gendered_celebs(g)
        data.extend([{"name": n, "gender": g} for n in names])

    return pd.DataFrame(data)


def get_celeb_popularity():
    path = f"/home/amor/Documents/code_dw/ai-pokemon-episode/external_data/celeb_names.txt"
    df = pd.read_csv(path, delimiter="\t", names=["name", "famous"])
    df["name"] = df["name"].apply(lambda x: x.replace("_", " "))
    return df


def gen_character_mix(df, gender):
    opposite_gender = "female" if gender == "male" else "male"
    age = random.randint(20, 40)
    selected = df[df.gender == opposite_gender].sample(n=2).name.values
    return "[{}]".format(" | ".join(selected)) + f" as a {age} year old {gender}"


def add_random_attributes(gender, n=1):
    attributes = ["hat", "glasses", "", "", "", ""]
    body_shape = ["plus sized", "fit", "slim", "", ""]

    clothes = ["clothes", "outfit", "tuxedo", "shirt", "suit"]
    if gender == "female":
        clothes += ["dress", "skirt",]

    colors = ["red", "blue", "green", "white", "black", "yellow"]

    actions = ["drinking", "action pose", "surprised", "smiling", "looking at the viewer", "reflexive pose",
               "flexing", "running"]

    prompt = ""
    for selected in [attributes, body_shape, colors, clothes, actions]:
        prompt += random.sample(selected, n)[0] + ", "

    return prompt


def single_tag(gender):
    return "1woman" if gender == "female" else "1man"


def get_random_char_prompt(gender):
    opposite_gender = "female" if gender == "male" else "male"
    df1 = build_celeb_df()
    df2 = get_celeb_popularity()
    df3 = df1.merge(df2, on="name").sort_values("famous", ascending=False)
    pos = gen_character_mix(df3, gender)

    outfit_kw = add_random_attributes(gender)

    pos_tags = [pos, outfit_kw, single_tag(gender)]
    neg_tags = [single_tag(opposite_gender)]

    return " ".join(pos_tags), " ".join(neg_tags)


# Define lightweight function for resolving references in text
def resolve_references(doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


def text_reference_resolver(text):
    nlp = spacy.load("en_coreference_web_trf")
    doc = nlp(text)
    return doc, {cluster: [e.text for e in doc.spans[cluster] if
                           len(e.text.split(" ")) >= 2 or len(e.text) >= 5]
                 for cluster in doc.spans}


def find_largest_coref_prompt_in_sentence(sentence, character_config):
    if character_config is None:
        return "", sentence

    matched = defaultdict(list)
    for name, config in character_config.items():
        references = config["text_parts"]
        for part in references:
            for sub_part in str(part.strip()).split(","):
                if sub_part and str(sub_part) in sentence:
                    matched[name].append(sub_part)

    if len(matched) == 0:
        return "", sentence
    reference = sorted(matched.keys(), key=lambda x: len(matched[x]), reverse=True)[0]

    print("\n")
    print(sentence, matched)
    print("\n")

    if len(matched) == 0:
        return "", sentence
    # surgery : remove matched eleemnt in sentence
    tokens = sentence.split(" ")
    kept = set(tokens).difference(set(matched[reference]))
    return character_config[reference]["prompt"], ", ".join(kept)
