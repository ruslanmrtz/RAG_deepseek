from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda")

embedding_model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
embedding_model.to(device)


def get_embedding(text, task = "search_document"):
    prefixed_text = f"{task}: {text}"
    # prefixed_text = text

    embedding = embedding_model.encode(
        prefixed_text,
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False
    ).tolist()
    return embedding


print(len(get_embedding('Привет как дела это новый кадилак')))


