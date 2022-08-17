from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Corpus with example sentences
base_path = os.path.join(args.data_root, 'lareqa')
dataset_to_dir = {
    "xquad": "xquad-r",
}

squad_dir = os.path.join(base_path, dataset_to_dir["xquad"])

# Load the question set and candidate set.
squad_per_lang = {}
languages = set()
# Load all files in the given directory, expecting names like 'en.json',
# 'es.json', etc.
for filename in os.listdir(squad_dir):
    language = os.path.splitext(filename)[0]
    languages.add(language)
    with open(os.path.join(squad_dir, filename), "r") as f:
    squad_per_lang[language] = json.load(f)
    print("Loaded %s" % filename)
question_set, candidate_set = utils_lareqa.load_data(squad_per_lang)

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))