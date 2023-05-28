import os
import pandas as pd
import random


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
    attributes = ["hat", "kepi", "", "", "", ""]
    body_shape = ["thicc", "fit", "slim", "", ""]

    clothes = ["clothes", "outfit", "tuxedo", "shirt", "suit"]
    if gender == "female":
        clothes += ["dress", "skirt",]

    colors = ["red", "blue", "green", "white", "black", "yellow"]

    prompt = ""
    for selected in [attributes, body_shape, colors, clothes]:
        prompt += random.sample(selected, n)[0] + " "

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

