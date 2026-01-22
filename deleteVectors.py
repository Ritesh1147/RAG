import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#import sqlite3
#sys.path.append('/opt/thelvwapp/Documents/MultiModelRAG/ver1.1')
#import IPython
#from IPython.display import HTML, display, Image, Markdown, Video, Audio
from typing import Optional, Sequence, List, Dict, Union

import soundfile as sf

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#from google.colab import userdata

from sentence_transformers import SentenceTransformer
from transformers import ClapModel, ClapProcessor
from datasets import load_dataset

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.api.types import Document, Embedding, EmbeddingFunction, URI, DataLoader

import numpy as np
import torchaudio
import base64
import torch
import json
import cv2
import os

#Modality 1 Audio retrieval
#def load_audio_dataset():
path = "mm_vdbChain"
client = chromadb.PersistentClient(path=path)
print(client)
#ids=client.get()['ids']
#print(ids)


CLIP = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

video_collectionBOJ = client.get_or_create_collection(
    name='video_collectionChain',
    embedding_function=CLIP,
    data_loader=image_loader
)

videoIds=video_collectionBOJ.get()['ids']
#print(ids)
if len(videoIds) > 0:
 video_collectionBOJ.delete(videoIds)

image_collectionBOJ = client.get_or_create_collection(name="image_collection",
                                                   embedding_function = CLIP,
                                                   data_loader = image_loader)
imageIds= image_collectionBOJ.get()['ids']
print(imageIds)
if len(imageIds) > 0:
 image_collectionBOJ.delete(imageIds)

text_collectionBOJ = client.get_or_create_collection(name="text_collection")
textIds= text_collectionBOJ.get()['ids']
print(textIds)
if len(textIds) > 0:
 for textid in textIds:
  text_collectionBOJ.delete(textid)


class AudioLoader(DataLoader[List[Optional[Dict[str, any]]]]):
    def __init__(self, target_sample_rate: int = 48000) -> None:
        self.target_sample_rate = target_sample_rate

    def _load_audio(self, uri: Optional[URI]) -> Optional[Dict[str, any]]:
        if uri is None:
            return None

        try:
            waveform, sample_rate = torchaudio.load(uri)

            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return {"waveform": waveform.squeeze(), "uri": uri}
        except Exception as e:
            print(f"Error loading audio file {uri}: {str(e)}")
            return None

    def __call__(self, uris: Sequence[Optional[URI]]) -> List[Optional[Dict[str, any]]]:
        return [self._load_audio(uri) for uri in uris]
    
class CLAPEmbeddingFunction(EmbeddingFunction[Union[Document, Dict[str, any]]]):
    def __init__(
        self,
        model_name: str = "laion/larger_clap_general",
        device: str = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.device = device

    def _encode_audio(self, audio: torch.Tensor) -> Embedding:
        inputs = self.processor(audios=audio.numpy(), sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_embedding = self.model.get_audio_features(**inputs)
        return audio_embedding.squeeze().cpu().numpy().tolist()

    def _encode_text(self, text: Document) -> Embedding:
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        return text_embedding.squeeze().cpu().numpy().tolist()

    def __call__(self, input: Union[List[Document], List[Optional[Dict[str, any]]]]) -> List[Optional[Embedding]]:
        embeddings = []
        for item in input:
            if isinstance(item, dict) and 'waveform' in item:
                embeddings.append(self._encode_audio(item['waveform']))
            elif isinstance(item, str):
                embeddings.append(self._encode_text(item))
            elif item is None:
                embeddings.append(None)
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return embeddings
    


audio_collection = client.get_or_create_collection(
   name='audio_collection',
   embedding_function=CLAPEmbeddingFunction(),
   data_loader=AudioLoader()
 )


audioIds= audio_collection.get()['ids']
print(audioIds)
if len(audioIds) > 0:
 audio_collection.delete(audioIds)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
   model_name=embeddings_model_path,
 )
# embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
 
os.environ["OPENAI_API_KEY"]="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"
# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql://postgres:postgres@localhost:5432/postgres"  # Uses psycopg3!
collection_name = "my_pdfdocsChain"
   #embeddings=OpenAIEmbeddings(model="ibm-granite/granite-embedding-30m-english"),#"text-embedding-3-large"),
 
vector_store = PGVector(
    embeddings=embeddings_model, 
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
 )
# documents = list(itertools.chain(texts, tables, pictures))
#doclingvectorids = vector_store.embeddings[textIds] 
print("doclingvector start")
#print(doclingvectorids)
#collections = vector_store.list_collections()
#print(collections)
vector_store.delete_collection()
print("doclingvector end")
#if len(doclingvectorids) > 0:
# vector_store.delete(doclingvectorids)

