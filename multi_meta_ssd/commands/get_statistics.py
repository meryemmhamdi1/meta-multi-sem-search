import json, pickle
from os import listdir
from os.path import isfile, join
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options

def get_statistics(subparser):
    parser = subparser.add_parser("get_stats", help="Get statistics")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    parser.set_defaults(func=run_stsb_stats)

def run_lareqa_stats(args):
    # for lang in ["ar", "de", "el", "hi", "ru", "th", "tr"]:
    #     with open("/sensei-fs/users/mhamdi/Datasets/meta-multi-sem-search/lareqa/xquad-r/cross_val/0/test/"+lang+".json") as file:
    #         data = json.load(file)

    #     print("len(data):", len(data['data'][0]))

    print("-------------------------------------------------------------")

    root_dir = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/meta_tasks/TripletLoss/BIL_MULTI/random/ar,de,el,hi,ru,th,tr/CrossVal_0/"
    languages = ["ar", "de", "el", "hi", "ru", "th", "tr"] 
    lang_dict = {"ar": "Arabic", "de": "German", "el": "Dutch", "hi": "Hindi", "ru": "Russian", "th": "Thai", "tr": " Turkish"}
    lang_lines = [lang_dict[lang] + " & " + lang.upper() for lang in languages]
    for split_name in ["train", "valid", "test"]:
        with open(root_dir+split_name+"_question_set.pickle", "rb") as file:
            question_set = pickle.load(file)

        with open(root_dir+split_name+"_candidate_set.pickle", "rb") as file:
            candidate_set = pickle.load(file)

        for i, lang in enumerate(languages):
            lang_lines[i] += " & " + str(len(question_set.by_lang[lang])) + " & " + str(len(candidate_set.by_lang[lang]))
            print("split_name:", split_name, " lang:", lang, " len(questions):", len(question_set.by_lang[lang]))
            print("split_name:", split_name, " lang:", lang, " len(candidates):", len(candidate_set.by_lang[lang]))

    print(lang_lines)

def run_tatoeba_stats(args):
    # mypath = "/sensei-fs/users/mhamdi/xtreme/download/tatoeba/"
    # mypath = "/sensei-fs/users/mhamdi/LASER/data/tatoeba/v1/"
    mypath = "/sensei-fs/users/mhamdi/xtreme/download/tatoeba_labels/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for filename in onlyfiles:
        with open(join(mypath, filename)) as file:
            data = file.readlines()
        
        print("filename:", filename, " len(data):", len(data))


def run_stsb_stats(args):
    mypath = "/sensei-fs/users/mhamdi/Datasets/STS2017/STS2017.eval.v1.1/"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for filename in onlyfiles:
        with open(join(mypath, filename)) as file:
            data = file.readlines()
        
        print("filename:", filename, " len(data):", len(data))