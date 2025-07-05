import pandas as pd
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from langchain_text_splitters import RecursiveCharacterTextSplitter

import RoSBERTa

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


client = QdrantClient(url="http://localhost:6333")

# language = 'russian'  # можно заменить на 'english'
# stop_words = set(stopwords.words(language))

try:
    client.create_collection(
        collection_name="success_collection",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("Collection created.")
except Exception as e:
    print('Таблица уже существует!')


def make_chunks(text):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=['\n\n', '\n', '.', '!', '?']
    )
    chunks = spliter.split_text(text)

    return chunks


def preprocess_data(text):

    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def add_point(text, payload):
    embedding = RoSBERTa.get_embedding(text)

    client.upsert(
        collection_name='success_collection',
        points=[
            PointStruct(
                id=str(uuid.uuid4().hex),
                vector=embedding,
                payload=payload
            )])


def get_data_from_json():
    with open('result.json', 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
        count_mes = len(data['messages'])
        print('Всего сообщений:', count_mes)
        for mes in tqdm(data['messages']):
            if mes['text_entities']:
                text = ''
                for chank in mes['text_entities']:
                    text += chank['text']

                chunks = make_chunks(text)
                preprocessed_text = preprocess_data(text)
                for chunk in chunks:
                    preprocessed_chunk = preprocess_data(chunk)

                    payload = {
                        'content': preprocessed_text,
                        'metadata': {
                            'message_id': mes['id']
                        }
                    }

                    add_point(preprocessed_chunk, payload)


def find_text(text):
    results = client.query_points(
        collection_name='success_collection',
        query=RoSBERTa.get_embedding(text.lower()),
        limit=5,
        with_payload=True
    )

    line = ''
    for _, scope in results:
        for point in scope:
            line += point.payload['content']
            line += ' '

    return line


if __name__ == '__main__':
    # get_data_from_json()

    print(find_text('Как заработать миллион молодому человеку'))
