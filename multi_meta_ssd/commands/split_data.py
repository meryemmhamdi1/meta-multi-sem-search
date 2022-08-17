import os, json, random
import numpy as np
import sys
sys.path.append("/Users/meryemmhamdi/Library/CloudStorage/OneDrive-Adobe/Adobe/MutiSemSearch/Code/meta-multi-sem-search")
import multi_meta_ssd.processors.downstream.utils_lareqa as utils_lareqa
from sklearn.model_selection import KFold, train_test_split

random.seed(2342556)

data_root = "/Users/meryemmhamdi/Library/CloudStorage/OneDrive-Adobe/Adobe/MutiSemSearch/Datasets/"
base_path = os.path.join(data_root, 'lareqa')
dataset_to_dir = {
    "xquad": "xquad-r",
    "mlqa": "mlqa-r"
}
squad_dir = os.path.join(base_path, dataset_to_dir["xquad"])
mlqar_dir = os.path.join(base_path, dataset_to_dir["mlqa"])

all_languages = "ar,de,el,hi,ru,th,tr"
mlqa_languages = "ar,de,en,es,hi,vi,zh"

def split_data_options(subparser):
    parser = subparser.add_parser("split_lareqa_cross_val", help="Split Lareqa using cross-validation")
    parser.set_defaults(func=split_cross_validation)

def read_lareqa():
    squad_per_lang = {}
    languages = set(all_languages.split(","))
    # Load all files in the given directory, expecting names like 'en.json',
    # 'es.json', etc.
    for filename in os.listdir(squad_dir):
        language = os.path.splitext(filename)[0]
        languages.add(language)
        if ".json" in filename:
            with open(os.path.join(squad_dir, filename), "r") as f:
                print("os.path.join(squad_dir, filename):", os.path.join(squad_dir, filename))
                squad_per_lang[language] = json.load(f)
        print("Loaded %s" % filename)

    return squad_per_lang

en_squad_dir = os.path.join(data_root, "squad")
def read_en_squad():
    # Load files from English (train and test)
    # print("en_squad_dir:", en_squad_dir)
    with open(os.path.join(en_squad_dir, "train-v1.1.json"), "r") as f:
        data_train = json.load(f)

    with open(os.path.join(en_squad_dir, "dev-v1.1.json"), "r") as f:
        data_dev = json.load(f)

    print("len(data_train):", len(data_train["data"]), " len(data_dev):", len(data_dev["data"]))

    return data_train, data_dev


def save_data(data, lang, save_dir, split, cross_val=False, i=None):
    # print("path:", os.path.join(save_dir,"splits", split))
    if cross_val:
        folder = "cross_val"
    else:
        folder = "splits"

    full_folder_path = os.path.join(save_dir, folder, str(i), split)
    if not os.path.isdir(full_folder_path):
        os.makedirs(full_folder_path)

    with open(os.path.join(full_folder_path, lang + ".json"), "w") as file:
        json.dump({"data": data, "version": "splits June 28 2022"}, file)

def split_lareqa(train_titles, valid_titles, test_titles):
    squad_per_lang = read_lareqa()

    train_data = {}
    valid_data = {}
    test_data = {}

    for lang in all_languages.split(","):
        train_data = []
        valid_data = []
        test_data = []
        for passage in squad_per_lang[lang]["data"]:
            if passage["title"] in train_titles: #, valid_titles, test_titles
                train_data.append(passage)
            elif passage["title"] in valid_titles:
                valid_data.append(passage)
            elif passage["title"] in test_titles:
                test_data.append(passage)

        print("total_count:", len(train_data)+len(valid_data)+len(test_data), " len(squad_per_lang[lang]):", len(squad_per_lang[lang]["data"]))

        save_data(train_data, lang, squad_dir, split="train")
        save_data(valid_data, lang, squad_dir, split="valid")
        save_data(test_data, lang, squad_dir, split="test")

    return train_titles, valid_titles, test_titles

def split_en_squad():
    data_train, data_dev = read_en_squad()

    # Splitting train portion of English SQUAD into train and validation
    # json_sz = len(data_train["data"])
    json_sz = len(data_dev["data"])
    num_train = int(json_sz * 0.70)
    num_valid = (json_sz - num_train)//2

    # train_data = data_train["data"][:num_train]
    # valid_data = data_train["data"][num_train:]
    # test_data = data_dev["data"]

    train_data = data_dev["data"][:num_train]
    valid_data = data_dev["data"][num_train:num_train+num_valid]
    test_data = data_dev["data"][num_train+num_valid:]

    train_titles = [train_data[i]["title"] for i in range(len(train_data))]
    valid_titles = [valid_data[i]["title"] for i in range(len(valid_data))]
    test_titles = [test_data[i]["title"] for i in range(len(test_data))]

    save_data(train_data, "en", squad_dir, split="train")
    save_data(valid_data, "en", squad_dir, split="valid")
    save_data(test_data, "en", squad_dir, split="test")

    return train_titles, valid_titles, test_titles

def split_mlqar():
    mlqa_per_lang = {}
    languages = set(mlqa_languages.split(","))
    # Load all files in the given directory, expecting names like 'en.json',
    # 'es.json', etc.
    for filename in os.listdir(mlqar_dir):
        language = os.path.splitext(filename)[0]
        languages.add(language)
        if ".json" in filename:
            with open(os.path.join(mlqar_dir, filename), "r") as f:
                mlqa_per_lang[language] = json.load(f)
            print("Loaded %s" % filename)


    question_set, _ = utils_lareqa.load_data(mlqa_per_lang)
    for lang in all_languages.split(","):
        print("lang:", lang, "question_set.by_lang:", len(question_set.by_lang[lang]))

    num_questions = 0
    counter_per_rank = {i:0 for i in range(8)}
    for xling_id in question_set.by_xling_id:
        num_questions += 1
        counter = 0
        for question in question_set.by_xling_id[xling_id]:
            # print("question.language:", question.language)
            counter += 1
        counter_per_rank[counter] += 1

    print("NUM OF QUESTIONS:", num_questions)
    print("counter_per_rank:", counter_per_rank)

    for lang in mlqa_languages.split(","):
        json_sz = len(mlqa_per_lang[lang]["data"])
        # print("lang:", lang, " json_sz:", json_sz)
        num_train = int(json_sz * 0.60)
        num_valid = (json_sz - num_train)//2

        # squad_per_lang_shuff = squad_per_lang[lang]["data"][indices]
        squad_per_lang_shuff = mlqa_per_lang
        train_data = squad_per_lang_shuff[lang]["data"][:num_train]
        valid_data = squad_per_lang_shuff[lang]["data"][num_train:num_train+num_valid]
        test_data = squad_per_lang_shuff[lang]["data"][num_train+num_valid:]

        save_data(train_data, lang, mlqar_dir, split="train")
        save_data(valid_data, lang, mlqar_dir, split="valid")
        save_data(test_data, lang, mlqar_dir, split="test")

def split_cross_validation(args):
    squad_per_lang = read_lareqa()

    ## Pick train and test tiles then divide train into train and validation
    train_data = {}
    valid_data = {}
    test_data = {}

    kf = KFold(n_splits=5)

    for lang in all_languages.split(","):
        # Get the list of all titles
        titles = [squad_per_lang[lang]["data"][i]["title"] for i in range(len(squad_per_lang[lang]["data"]))]
        print("lang:", lang, " len(titles):", len(titles))
        i = 0

        for train_index, test_index in kf.split(titles):
            print("train_index:", train_index, " test_index:", test_index)

            train_valid_titles, test_titles = [titles[idx] for idx in train_index], [titles[idx] for idx in test_index]

            num_valid = len(test_titles)

            train_titles = train_valid_titles[:len(train_valid_titles) - num_valid]
            valid_titles = train_valid_titles[len(train_valid_titles) - num_valid:]

            # Divide them here
            train_data = []
            valid_data = []
            test_data = []
            for passage in squad_per_lang[lang]["data"]:
                if passage["title"] in train_titles: #, valid_titles, test_titles
                    train_data.append(passage)
                elif passage["title"] in valid_titles:
                    valid_data.append(passage)
                elif passage["title"] in test_titles:
                    test_data.append(passage)

            print("total_count:", len(train_data)+len(valid_data)+len(test_data), " len(squad_per_lang[lang]):", len(squad_per_lang[lang]["data"]))

            save_data(train_data, lang, squad_dir, split="train", cross_val=True, i=i)
            save_data(valid_data, lang, squad_dir, split="valid", cross_val=True, i=i)
            save_data(test_data, lang, squad_dir, split="test", cross_val=True, i=i)
            i += 1

    return train_titles, valid_titles, test_titles


# train_titles, valid_titles, test_titles = split_en_squad()
# split_lareqa(train_titles, valid_titles, test_titles)
# split_mlqar()
# split_cross_validation()