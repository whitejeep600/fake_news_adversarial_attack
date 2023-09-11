import numpy as np
from transformers import AutoTokenizer, BertModel


def cosine_similarity(e1, e2):
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))


def get_embedding(word, model, tokenizer):
    word_id = tokenizer(word)["input_ids"]
    tensor = model.embeddings.word_embeddings.weight[word_id]
    return tensor.detach().numpy()


def main():
    model_name = "prajjwal1/bert-tiny"
    model = BertModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_words = [w for w in model.tokenizer.vocab if w.isalpha()]
    embeddings = np.array([get_embedding(word, model, tokenizer) for word in all_words])
    # get cosine similarities (vectorized)
    # for each word extract the 200 most similar words
    # save
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)


if __name__ == '__main__':
    main()
