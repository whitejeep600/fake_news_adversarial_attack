import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer, BertModel


def get_embedding(word, model, tokenizer):
    word_id = tokenizer(word)["input_ids"][1]
    tensor = model.embeddings.word_embeddings.weight[word_id]
    return tensor.detach().numpy()


def main(n_neighbors_precomputed: int, neighbors_path: Path):
    model_name = "prajjwal1/bert-tiny"
    model = BertModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_words = [w for w in tokenizer.vocab if w.isalpha()]
    embeddings = np.array([get_embedding(word, model, tokenizer) for word in all_words])

    dot_products = embeddings @ embeddings.T
    norms = np.linalg.norm(embeddings, axis=1)
    cosine_similarities = dot_products / norms[None, :] / norms[:, None]

    assert np.allclose(np.diag(cosine_similarities), np.ones(len(all_words)), 0.001)

    # we don't want to extract a word as its own neighbor
    np.fill_diagonal(cosine_similarities, 0)
    similar_word_inds = np.argsort(-cosine_similarities)[:, :n_neighbors_precomputed]
    neighbors = {
        all_words[i]: [all_words[similar_word_inds[i][j]] for j in range(n_neighbors_precomputed)]
        for i in range(len(all_words))
    }

    with open(neighbors_path, "w") as f:
        f.write(json.dumps(neighbors, indent=2))


if __name__ == "__main__":
    text_fooler_params = yaml.safe_load(open("params.yaml"))["attacks.text_fooler"]
    n_neighbors_precomputed = int(text_fooler_params["n_neighbors_precomputed"])
    neighbors_path = Path(text_fooler_params["neighbors_path"])
    main(n_neighbors_precomputed, neighbors_path)
