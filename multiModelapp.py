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

audio_folder_path="esc50"

#Modality 1 Audio retrieval
def load_audio_dataset():
 path = "mm_vdbChain"
 client = chromadb.PersistentClient(path=path)
 #ds = load_dataset("ashraq/esc50")

 #ds =  load_dataset("audiofolder", data_dir="/home/mkandra/Documents/BOJ/BOJMultiModelRAG/InpuTAudioFiles")
 ds =  load_dataset("audiofolder", data_dir="InpuTAudioFiles")



# Define the directory to save audio files
 path = "esc50"
 os.makedirs(path, exist_ok=True)

# Process and save audio files
 for item in ds['train']:
    audio_array = item['audio']['array']
    sample_rate = item['audio']['sampling_rate']
    file_name = item['filename']
    target_path = os.path.join(path, file_name)

    # Write the audio file to the new directory
    sf.write(target_path, audio_array, sample_rate)

 print("All audio files have been processed and saved.")
 return client

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
    


# Takes a couple mins with GPU
def add_audio(audio_collection, folder_path):
    # List to store IDs and URIs
    ids = []
    uris = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_id = os.path.splitext(filename)[0]
            file_uri = os.path.join(folder_path, filename)
            print("file id")
            print(file_id)
            print("file uri")
            print(file_uri)
            ids.append(file_id)
            uris.append(file_uri)

    # Add files to the collection
    audio_collection.add(ids=ids, uris=uris)


import shutil
def copy_files_static(file_paths):
 files_shown = 0
 file_name1=""
 file_name2=""
 file_name3=""
 file_name4=""
 file_name5=""
#plt.figure(figsize=(16, 9))
 static_files=[]
 for file_path in file_paths:
# print(img_path)
  if os.path.isfile(file_path):
   head, file_name = os.path.split(file_path)
   print("file_name")
   print(file_name)
#  dest_path = "/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/static/" + file_name
  dest_path = "static/" + file_name
  shutil.copy(file_path,dest_path)
  files_shown += 1
  if files_shown == 1:
   file_name1= "/"+file_name
   static_files.append(file_name1)
  elif files_shown == 2:
   file_name2= "/"+file_name
   static_files.append(file_name2)
  elif files_shown == 3:
   file_name3= "/"+file_name
   static_files.append(file_name3)
  elif files_shown == 4:
   file_name4= "/"+file_name
   static_files.append(file_name4)
  elif files_shown == 5:
   file_name5= "/"+file_name
   static_files.append(file_name5)
  if files_shown >= 6:
   break
 return static_files


def copy_file_to_static(file_path):
 files_shown = 0
 static_files=[]
 if os.path.isfile(file_path):
   head, file_name = os.path.split(file_path)
   print("file_name")
   print(file_name)
   #dest_path = "/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/static/" + file_name
   dest_path = "static/" + file_name
   shutil.copy(file_path,dest_path)
   files_shown += 1
   dest_file=""
   if files_shown == 1:
    dest_file= "/"+file_name
   return dest_file


# Running it
def load_audio_embeddings(client):
  #client=load_audio_dataset()
  audio_collection = client.get_or_create_collection(
   name='audio_collectionChain',
   embedding_function=CLAPEmbeddingFunction(),
   data_loader=AudioLoader()
 )
  folder_path = 'esc50'
  add_audio(audio_collection, folder_path)
  print("add audio collection completed")
  return audio_collection

	




from IPython.display import Audio

def display_audio_files(query_text, max_distance=None, debug=False):
    # Query the audio collection with the specified text
    results = audio_collection.query(
        query_texts=[query_text],
        n_results=3,
        include=['uris', 'distances']
    )

    # Extract uris and distances from the result
    uris = results['uris'][0]
    distances = results['distances'][0]

    # Display the audio files that meet the distance criteria
    for uri, distance in zip(uris, distances):
        # Check if a max_distance filter is applied and the distance is within the allowed range
        if max_distance is None or distance <= max_distance:
            if debug:
              print(f"URI: {uri} - Distance: {distance}")
   #         display(Audio(uri))
              Audio(uri)
        else:
            if debug:
              print(f"URI: {uri} - Distance: {distance} (Filtered out)")

#    for file_instatic in os.listdir('/opt/thelvwapp/Documents/MultiModelRAG/doclingver1.1/static'):
#      if os.path.isfile(file_instatic):
#        os.remove(file_instatic)

    file_names = copy_files_static(uris)
    return file_names
# Running it



#Modality 2 Image retrival
def load_images():
 #inputFolder = '/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/InputImages'
 inputFolder = 'InputImages'
 #ds = load_dataset("KoalaAI/StockImages-CC0")

 ds = load_dataset("imagefolder",data_dir=inputFolder)

# Indices to remove
 indices_to_remove = {586, 1002}

# Generate a list of indices excluding the problematic ones
 all_indices = set(range(len(ds['train'])))
 indices_to_keep = list(all_indices - indices_to_remove)
# Select the remaining entries in the dataset
 ds['train'] = ds['train'].select(indices_to_keep)

# Verification
 print(ds['train'])
 return ds

output_folder = "StockImages-cc0"
os.makedirs(output_folder, exist_ok=True)

def process_and_save_image(idx, item):
    try:
        # Since the image is already a PIL image, just save it directly
        image = item['image']
        image.save(os.path.join(output_folder, f"image_{idx}.jpg"))
    except Exception:
        pass

def process_images(dataset):
    for idx, item in enumerate(dataset['train']):
        process_and_save_image(idx, item)

# Running it
def load_process_images(client,image_loader,CLIP):
 ds=load_images()
 process_images(ds)


# Instantiate the Image Loader
# image_loader = ImageLoader()
# Instantiate CLIP embeddings
# CLIP = OpenCLIPEmbeddingFunction()

# Create the image collection
 image_collection = client.get_or_create_collection(name="image_collectionChain",
                                                   embedding_function = CLIP,
                                                   data_loader = image_loader)

# Initialize lists for ids and uris
 ids = []
 uris = []

 dataset_folder="StockImages-cc0"
 
# Iterate over each file in the dataset folder
 for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.jpg'):
        file_path = os.path.join(dataset_folder, filename)

        # Append id and uri to respective lists
        ids.append(str(i))
        uris.append(file_path)

# Add to image collection
 image_collection.add(
    ids=ids,
    uris=uris
 )
 
 print("Images added to the database.")
 return image_collection

def display_images(image_collection, query_text, max_distance=None, debug=False):
    # Query the image collection with the specified text
    results = image_collection.query(
        query_texts=[query_text],
        n_results=5,
        include=['uris', 'distances']
    )

    # Extract uris and distances from the result
    uris = results['uris'][0]
    distances = results['distances'][0]

    # Display the images that meet the distance criteria
    for uri, distance in zip(uris, distances):
        # Check if a max_distance filter is applied and the distance is within the allowed range
        if max_distance is None or distance <= max_distance:
            if debug:
              print(f"URI: {uri} - Distance: {distance}")
        #    display(Image(uri, width=300))
        else:
            if debug:
              print(f"URI: {uri} - Distance: {distance} (Filtered out)")
   


    file_names = copy_files_static(uris)
    return file_names


# Running it
def query_response(image_collection,question, client):
 #for file_instatic in os.listdir('/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/static'):
 for file_instatic in os.listdir('static'):     
      if os.path.isfile(file_instatic):
        os.remove(file_instatic)
 audio_uris=display_audio_files(question, max_distance= 1.5, debug=True)
 image_uris=display_images(image_collection,question, max_distance= 1.5, debug=True)
 text_reponse,distances,metadatas, titles = display_text_documents(question, text_collection,max_distance=1.3, debug=True)
 video_uris = display_videos(question, max_distance= 1.55, debug=True)
 return audio_uris, image_uris, text_reponse,distances,metadatas, titles, video_uris



import speech_recognition as sr

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source, duration=300)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

from openai import OpenAI
api_key="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"

def save_audio_to_text_files(audio_folder_path, text_file_path):
 client = OpenAI(api_key=api_key)
 for filename in os.listdir(audio_folder_path):
        if filename.endswith('.wav'):
         file_uri = os.path.join(audio_folder_path, filename)
         audio_file= open(file_uri, "rb")
         transcription = client.audio.transcriptions.create(
        # model="gpt-4o-transcribe",
         model="gpt-4o-mini-transcribe" ,
        # model="whisper-1", 
         file=audio_file
         )
         head, filebasename = os.path.split(file_uri)
         text_file_base_name, extension = os.path.splitext(filebasename)
         text_filename=os.path.join(text_file_path, text_file_base_name + ".txt")
         with open(text_filename, "w") as f:
           f.write(transcription.text)
           print("text file name")
           print(text_filename)





def textDataload(client):
   #inputFolder="/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/inputTextFiles"
   inputFolder="inputTextFiles"
   #ds = load_dataset("TopicNavi/Wikipedia-example-data")

   save_audio_to_text_files(audio_folder_path,inputFolder)

   ds = load_dataset("text", data_dir=inputFolder)
   print("ds -------------start")
   print(ds)
   print("ds ---------end")
   # Check if CUDA is available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")

# Load the model
   model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
   
    #sentences = text.split(". "

# Prepare the data
   documents = [entry['text'] for entry in ds['train']]
   #metadatas = [{"url": entry['url'], "wiki_id": entry['wiki_id']} for entry in ds['train']]
   #ids = [entry['title'] for entry in ds['train']]
   #ids = [entry['label'] for entry in ds['train']]
    
    
   print("documents len") 
   print(len(documents))
# Generate embeddings
   embeddings = []
   ids = []
   batch_size =  128  #128  # Adjustable based GPU memory


   for i in range(0, len(documents)):
     ids.append(str(i))


   for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
    embeddings.extend(batch_embeddings.cpu().numpy())
    
# Convert embeddings to list for JSON serialization        emb_batch = embeddings[i]
    print("embeddings count before")
    print(len(embeddings))


   embeddings = [emb.tolist() for emb in embeddings]

   print("embeddings count")
   print(len(embeddings))

# Prepare the data for exportall-MiniLM-L6-v2
   export_data = {
    "documents": documents,
    "embeddings": embeddings,
  #  "metadatas": metadatas,
    "ids": ids
    }

# Export
   with open('wikipedia_embeddings.json', 'w') as f:
     json.dump(export_data, f)

   print("Data exported to wikipedia_embeddings.json")
   text_collection = client.get_or_create_collection(name="text_collectionChain")
   return ds, text_collection


def load_wiki(file_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract the components
    documents = data['documents']
    embeddings = data['embeddings']
   # metadatas = data['metadatas']
    ids = data['ids']

    print(f"Loaded data from {file_path}")
    print(f"Number of entries: {len(documents)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    #return documents, embeddings, metadatas, ids
    return documents, embeddings  ,  ids

 #Batch and add data to the collection, respecting max batch size
##def batch_add_to_collection(collection, documents, embeddings, metadatas, ids, batch_size=5461):
def batch_add_to_collection(collection, documents, embeddings,ids, batch_size=5461):
    print("documents len batch add")
    print(len(documents))
    for i in range(0, len(documents), batch_size):
        # Slice the data into batches
        doc_batch = documents[i:i + batch_size]
        emb_batch = embeddings[i:i + batch_size]
#        meta_batch = metadatas[i:i + batch_size]
        id_batch = ids[i:i + batch_size]
        print("doc_batch")
        print(len(doc_batch))
        print("emb_batch")
        print(len(emb_batch))
        print("meta_batch")
 #       print(len(meta_batch))
        print("id Batch") 
        print(len(id_batch))

        

        # Add the batch to the collection
        collection.add(
            documents=doc_batch,
            embeddings=emb_batch,
        #    metadatas=meta_batch,
            ids=id_batch
        )
        print(f"Batch {i // batch_size + 1} added to the collection successfully.")

def display_text_documents(query_text,text_collection, max_distance=None, debug=False):
    # Query the text collection with the specified text
    results = text_collection.query(
        query_texts=[query_text],
        n_results=5,
        #include=['documents', 'distances', 'metadatas']
        include=['documents', 'distances']
    )

    documents = results['documents'][0]
    distances = results['distances'][0]
  #  metadatas = results['metadatas'][0]
  #  titles = results['ids'][0]

    # Display the text documents that meet the distance criteria or are filtered out
 #   for title, doc, distance, metadata in zip(titles, documents, distances, metadatas):
   # for title, doc, distance in zip(titles, documents, distances):
    for doc, distance in zip( documents, distances):
   
       #    url = metadata.get('url')

        if max_distance is None or distance <= max_distance:
#            print(f"Title: {title.replace('_', ' ')}")
            if debug:
              print(f"Distance: {distance}")
    #        print(f"URL: {url}")
            print(f"Text: {doc}\n")
        else:
            # Output filtered out documents with their title and distance
            if debug:
              print(f" Distance: {distance} (Filtered out)")
  #             print(f"Title: {title.replace('_', ' ')} - Distance: {distance} (Filtered out)")
    #return documents,distances,metadatas, titles
    #return documents,distances, titles
    return documents,distances

def extract_frames(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_filename in os.listdir(video_folder):
        if video_filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_filename)
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            output_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0])
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            success, image = video_capture.read()
            frame_number = 0
            while success:
                if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
                    frame_time = frame_number / fps
                    output_frame_filename = os.path.join(output_subfolder, f'frame_{int(frame_time)}.jpg')
                    cv2.imwrite(output_frame_filename, image)

                success, image = video_capture.read()
                frame_number += 1

            video_capture.release()

def add_frames_to_chromadb(video_dir, frames_dir):
    # Dictionary to hold video titles and their corresponding frames
    video_frames = {}

    # Process each video and associate its frames
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_title = video_file[:-4]
            frame_folder = os.path.join(frames_dir, video_title)
            if os.path.exists(frame_folder):
                # List all jpg files in the folder
                video_frames[video_title] = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]

    # Prepare ids, uris and metadatas
    ids = []
    uris = []
    metadatas = []

    for video_title, frames in video_frames.items():
        video_path = os.path.join(video_dir, f"{video_title}.mp4")
        for frame in frames:
            frame_id = f"{frame[:-4]}_{video_title}"
            frame_path = os.path.join(frames_dir, video_title, frame)
            ids.append(frame_id)
            uris.append(frame_path)
            metadatas.append({'video_uri': video_path})

    video_collectionBOJ.add(ids=ids, uris=uris, metadatas=metadatas)

def display_videos(query_text, max_distance=None, max_results=5, debug=False):
    # Deduplication set
    displayed_videos = set()

    # Query the video collection with the specified text
    results = video_collectionBOJ.query(
        query_texts=[query_text],
        n_results=max_results,  # Adjust the number of results if needed
        include=['uris', 'distances', 'metadatas']
    )
    video_uris=[]
    # Extract URIs, distances, and metadatas from the result
    uris = results['uris'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    # Display the videos that meet the distance criteria
    for uri, distance, metadata in zip(uris, distances, metadatas):
        video_uri = metadata['video_uri']

        video_uris.append(video_uri) 

        # Check if a max_distance filter is applied and the distance is within the allowed range
        if (max_distance is None or distance <= max_distance) and video_uri not in displayed_videos:
            if debug:
              print(f"URI: {uri} - Video URI: {video_uri} - Distance: {distance}")
#            display(Video(video_uri, embed=True, width=300))
            displayed_videos.add(video_uri)  # Add to the set to prevent duplication
        else:
            if debug:
              print(f"URI: {uri} - Video URI: {video_uri} - Distance: {distance} (Filtered out)")
    
    file_names = copy_files_static(video_uris)
    return file_names

def image_uris(query_text, max_distance=None, max_results=5):
    results = image_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            filtered_uris.append(uri)

    return filtered_uris

def text_uris(query_text, max_distance=None, max_results=5):
    results = text_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['documents', 'distances']
    )

    filtered_texts = []
    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            filtered_texts.append(doc)

    return filtered_texts

def frame_uris(video_collection,query_text, max_distance=None, max_results=5):
    results = video_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    seen_folders = set()

    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            folder = os.path.dirname(uri)
            if folder not in seen_folders:
                filtered_uris.append(uri)
                seen_folders.add(folder)

        if len(filtered_uris) == max_results:
            break

    return filtered_uris



#audio_collection, client = load_audio_embeddings()
#client=load_audio_dataset()
path = "mm_vdbChain"
client = chromadb.PersistentClient(path=path)

question="dog"
audio_collection = load_audio_embeddings(client)




# Instantiate the Image Loader
image_loader = ImageLoader()
# Instantiate CLIP embeddings
CLIP = OpenCLIPEmbeddingFunction()
image_collection = load_process_images(client,image_loader,CLIP)

ds,text_collection=textDataload(client)
#print(ds)
#docs, embs, metas, ids = load_wiki('wikipedia_embeddings.json')
docs, embs ,ids = load_wiki('wikipedia_embeddings.json')

print(len(docs))
print(len(embs))
#text_collection = client.get_or_create_collection(name="text_collection")
#batch_add_to_collection(text_collection, docs, embs, metas, ids)
batch_add_to_collection(text_collection, docs, embs, ids)

# Running it
#display_text_documents("dog", text_collection,max_distance=1.3, debug=True)


 

video_folder_path = 'StockVideos-CC0'
output_folder_path = 'StockVideos-CC0-frames'

extract_frames(video_folder_path, output_folder_path)

video_collectionBOJ = client.get_or_create_collection(
    name='video_collectionChain',
    embedding_function=CLIP,
    data_loader=image_loader
)

# Running it
video_dir = 'StockVideos-CC0'
frames_dir = 'StockVideos-CC0-frames'

add_frames_to_chromadb(video_dir, frames_dir)


# Example usage:
images = image_uris("water droplet", max_distance=1.5)
#print("images")
#print(images)


# Example usage:
texts = text_uris("water", max_distance=1.3)
#print("texts")
#print(texts)

# Example usage:
vid_uris = frame_uris(video_collectionBOJ,"Trees", max_distance=1.55)
#print(vid_uris)

def define_LLM():
 api_key="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"
#api_key = userdata.get('OPENAI_API_KEY')

# Instantiate the LLM
 gpt4o = ChatOpenAI(model="gpt-4o", temperature = 0.0, api_key=api_key)

# Instantiate the Output Parserreturn documents,distances,metadatas, titles
 parser = StrOutputParser()

# Define the Prompt
 prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are document retrieval assistant that neatly synthesizes and explains the text and images provided by the user from the query {query}"),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "{texts}"
                },
                {
                    "type": "image_url",
                    "image_url": {'url': "data:image/jpeg;base64,{image_data_1}"}
                },             
                {
                    "type": "text",
                    "text": "This is a frame from a video, refer to it as a video:"
                },
                {
                    "type": "image_url",
                    "image_url": {'url': "data:image/jpeg;base64,{image_data_2}"}
                },

            ],
        ),
    ]
)
 '''
   {
                    "type": "image_url",
                    "image_url": {'url': "data:image/jpeg;base64,{image_data_1}"}
                },
'''
 chain = prompt | gpt4o | parser
 return chain
  
 
{
  "query": "the user query",
  "texts": "the retrieved texts",
  "image_data_1": "The retrieved image, base64 encoded",
  "image_data_2": "The retrieved frame, base64 encoded",
}
#  "image_data_1": "The retrieved image, base64 encoded",
chain=define_LLM()

def format_prompt_inputs(user_query):
   print("format begin")
   if len(frame_uris(video_collectionBOJ,user_query, max_distance=1.55))==0:
    frame=image_uris("Bank of Jamaica", max_distance=1.5)[0]
   else:    
    frame = frame_uris(video_collectionBOJ,user_query, max_distance=1.55)[0]
   print(frame)
   image=[]
   ###Image exclude   start
   if  len(image_uris(user_query, max_distance=1.5))==0:
    image=image_uris("Bank of Jamaica", max_distance=1.5)[0]
 
   else:
    image = image_uris(user_query, max_distance=1.5)[0]
   print(image)
   ##Image exclude end 
   if text_uris(user_query, max_distance=1.3)==0:
       text=[]
   else:
     text = text_uris(user_query, max_distance=1.3)
   print("text")  
   print(text)
   inputs = {}
    # save the user query
   inputs['query'] = user_query

    # Insert Text
   inputs['texts'] = text
   ##Image exclude ---start
   if len(image) > 0:
    # Encode the first image
     with open(image, 'rb') as image_file:
        image_data_1 = image_file.read()
        inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')
        print( " inputs['image_data_1'] ") 
   else:
       inputs['image_data_1'] =''
   ###Image exclude end
   if len(frame) > 0:
    # Encode the Frame
     with open(frame, 'rb') as image_file:
        image_data_2 = image_file.read()
        inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')
        print( " inputs['image_data_2'] ") 
   else:
    
       inputs['image_data_2']=''
    
  # if len(image)> 0:
   #    image = copy_file_to_static(image)
   #if len(frame) > 0:
    #    frame = copy_file_to_static(frame)   
   return frame, image, inputs

#query = "San Francisco"
query=""
import doclingapp
import datetime
from noteEngine import save_notes

def LLMresponse_for_query(query):
 #for file_instatic in os.listdir('/home/mkandra/Documents/BOJ/BOJMultiModelRAGChain/static'):
 for file_instatic in os.listdir('static'):
      if os.path.isfile(file_instatic):
        os.remove(file_instatic)
 frame, image, inputs = format_prompt_inputs(query)

 docling_answer,docling_page_content,docling_source, docling_chartNos,docling_pictures,docling_document,docling_responsePictures,docling_tupleResponse = doclingapp.query_vectordb(doclingapp.vector_db,doclingapp.documents, doclingapp.ids,  query, doclingapp.picture)
 responseChain = query_vectordbChain(doclingapp.vector_db, query)


 print("docling answer")
 print(docling_answer)
 print("Response Chain")
 print(responseChain)

 print("inputs bfore invoke")
 responseText= inputs['texts']
 print("response text...")
 print(responseText)
 #print(inputs)  
 if inputs !=None:
   responseTextLLM = chain.invoke(inputs)
 else:
   responseTextLLM=""
 print("response text 1")
 print(responseTextLLM)

 if len(frame) > 0:
  video = f"StockVideos-CC0/{frame.split('/')[1]}.mp4"
 else:
    video=[]

 audioFiles=display_audio_files(query, max_distance=1.2)
 print("image response")
 #print(image)
 print("frame response")
 #print(frame)
 print("video response")
 #print(video)
 #print("text response")
 #print(responseText)
 print("audio File[0]")
 print(audioFiles[0])
 head, filebasename = os.path.split(audioFiles[0])
 print("File Base Name")
 print(filebasename)
 video_filename, extension = os.path.splitext(filebasename)
 print("video file name")
 print(video_filename)
 video=os.path.join(video_folder_path,video_filename + ".mp4")
 print("video")

 if len(video) > 0:
  videoFile = copy_file_to_static(video)
 else:
    videoFile=[]
 if len(image) > 0:
  imageFile = copy_file_to_static(image)
 else:
    imageFile=[]
 if len(frame) > 0:
  frameFile = copy_file_to_static(frame)
 else:
    frameFile=[]

 
 currentTime = datetime.datetime.now()

 currentTimeStr =  currentTime.strftime('%m/%d/%Y %H:%M:%S')
 
 
 notes= currentTimeStr + os.linesep + str("Question:") + query + os.linesep +  str("Response:") + os.linesep + str("LLM:ibm-granite/granite-embedding-30m-english" )  +  os.linesep + docling_answer + os.linesep +str("LLM:gpt-4o") + os.linesep +  responseTextLLM + os.linesep + str("---------------------------------------------------------------" )  + os.linesep

 save_notes(notes)


 return imageFile,frameFile, videoFile, responseTextLLM,audioFiles,docling_answer,docling_page_content,docling_source, docling_chartNos,docling_pictures,docling_document,docling_responsePictures,docling_tupleResponse,responseChain 

Replicate_API_Key = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"

os.environ["REPLICATE_API_TOKEN"] = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"

from langchain_community.llms import Replicate
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
import PIL.Image
import PIL.ImageOps

from typing import Iterable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document as LCDocument
from langchain_core.runnables import RunnableParallel

#def format_docs(docs: Iterable[LCDocument]):
#    return "\n\n".join(doc.page_content for doc in docs)

def format_docs(d):
            return str(d.input)


def query_vectordbChain(vector_db, question):
 
 #defining the model for llm


 model_path = "ibm-granite/granite-3.2-8b-instruct"
 model = Replicate(
    model=model_path,
    replicate_api_token=Replicate_API_Key,
    model_kwargs={
        "max_tokens": 1000, # Set the maximum number of tokens to generate as output.
        "min_tokens": 100, # Set the minimum number of tokens to generate as output.
    },
 )


 tokenizer = AutoTokenizer.from_pretrained(model_path)


# creating the prompt for the query
 # Create a Granite prompt for question-answering with the retrieved context
 prompt = tokenizer.apply_chat_template(
    conversation=[{
        "role": "user",
        "content": "{input}",
    }],
    documents=[{
        "title": "placeholder",
        "text": "{context}",
    }],

    add_generation_prompt=True,
    tokenize=False,
 )
 prompt_template = PromptTemplate.from_template(template=prompt)

# Create a Granite document prompt template to wrap each retrieved document
 document_prompt_template = PromptTemplate.from_template(template="""\
 Document {doc_id}
 {page_content}""")
 document_separator="\n\n"

# Assemble the retrieChart 1.2val-augmented generation chain
 combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator,
 ) 

 '''# creating the ragchain for the query
 rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
 ) 
 '''





 tokenizer = AutoTokenizer.from_pretrained(model_path)
 #docs=vector_db.as_retriever().invoke(query)


 retriever = vector_db.as_retriever()

 prompt = PromptTemplate.from_template("""
 Context information is below.
---------------------
 {context}
 ---------------------
 Given the context information and not prior knowledge, answer the query.
 Query: {question}
 Answer:
 """)

 '''
    # Create a RAG chain that consists of multiple runnables
 rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))  # Assign the formatted context to the 'context' key
        | prompt  # Use the prompt template to generate a prompt
        | model  # Use the Ollama language model to generate an answer
        | StrOutputParser()  # Parse the output of the language model as a string
    )

    # Create a parallel runnable that runs the retriever and the question in parallel
 rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}  # Assign the retriever and the question to the respective keys
    ).assign(answer=rag_chain_from_docs)

 
 '''
 template = """Answer the question based only on the following context:
 {context}

 Question: {question}
 """

 # The prompt expects input with keys for "context" and "question"
 prompt = ChatPromptTemplate.from_template(template)

 retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
 )

 responseChain=retrieval_chain.invoke(question)
 print("responce chain")
 print(responseChain)
 '''
 from langchain.chains.retrieval import create_retrieval_chain
 from langchain.chains.combine_documents import create_stuff_documents_chain

 for doc in vector_db.as_retriever().invoke(query):
    print("Query Response - ")
    print(doc)
 #   image = picture.get_image(doc)
 #   image.show()
 #   print("Image:")
 #   display(image)
    print("=" * 80) # Separator for clarity
   #------------------------------








# creating the prompt for the query
 # Create a Granite prompt for question-answering with the retrieved context
 prompt = tokenizer.apply_chat_template(
    conversation=[{
        "role": "user",
        "content": "{input}",
    }],
    documents=[{
        "title": "placeholder",
        "text": "{context}",
    }],
    add_generation_prompt=True,
    tokenize=False,
 )
 prompt_template = PromptTemplate.from_template(template=prompt)

# Create a Granite document prompt template to wrap each retrieved document
 document_prompt_template = PromptTemplate.from_template(template="""\
 Document {doc_id}
 {page_content}""")
 document_separator="\n\n"

# Assemble the retrieChart 1.2val-augmented generation chain
 combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator,
 ) 

 # creating the ragchain for the query
 
 rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
) 
 
#getting the response from the rag for query
 from docling_core.types.doc.document import PictureDescriptionData
 from IPython import display
 html_buffer = []
 outputs = rag_chain.invoke({"input": query},  search_kwargs={"k": 2})
 print("output before")

 #pic = picture.get_image(outputs)
 print("Image before:")
 #pic.show()Chart 1.2 GOJ Global Bond Yields
 print("Image after:")
  #ref = picture.get_ref().cref
  # print(ref)

 print(outputs)
 answer = outputs['answer']
 print("Answer")
 print(answer)
 chartNos = doclingapp.get_chart_no_from_text(answer)
 print("Chart Nos")
 print(chartNos)
 page_content = outputs['context']
 source = outputs['input']
 retrievedDocs= page_content
 responsePictures=[] 
 i=0
 Picture1=""
 responseCount=0
 tupleResponse=[]
 respPic = PIL.Image



 for retrieveDoc in retrievedDocs:

   answer_docid=retrievedDocs[i].metadata['doc_id']
   source=retrievedDocs[i].metadata['source']
   response_pagecontent=retrievedDocs[i].page_content
   tupleItem=(answer_docid,source,response_pagecontent)
   tupleResponse.append(tupleItem)

   i += 1
   match,responsePic = doclingapp.get_picture(pictures,page_content,answer_docid,doclingapp.conversions)
   print("responsePic")
   print(responsePic)
   print(match)
   if match=="Yes":
#    responsePic.show()
    responseCount=responseCount+1
 #    respPic=PIL.Image.open(responsePic)
    picName="Picture"+str(responseCount)+".png"
    responsePictures.append('/'+picName)
    #responsePic.save("/home/mkandra/Documents/BOJ/BOJMultiModelRAG/static/"+picName)
    responsePic.save("static/"+picName)
    print(picName)
   if responseCount >= 1:
     Picture1=responsePictures[0]
    #print("response Picture")
    #print(responsePictures[0])

 print("tupleresponseItem")
 print(tupleResponse)
 '''
 #return answer,page_content,source,chartNos, pictures,outputs, responsePictures,tupleResponse
 return responseChain
