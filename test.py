#from doclingapp import get_image_text

from PIL import Image
import PIL.ImageOps

import io

import base64

import os

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
from transformers import AutoProcessor

import itertools
from docling_core.types.doc.document import RefItem

from langchain_core.documents import Document

os.environ["REPLICATE_API_TOKEN"] = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"


from gtts import gTTS
import os


def text_to_audio(text, audio_file):
    language='en'
    audio_file= "static/" + audio_file
    tts_file = gTTS(text=text, lang=language, slow=False)
    print("audio file")
    tts_file.save(audio_file)


def encode_image(image: PIL.Image.Image, format: str = "png") -> str:
    image = PIL.ImageOps.exif_transpose(image) or image
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format)
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uri = f"data:image/{format};base64,{encoding}"
    return uri

Replicate_API_Key_Image = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"



def get_image_text_info(imagePath):
 
 imageFilePath= "static/" + imagePath 

 print("image path")
 print(imageFilePath)

 image=Image.open(imageFilePath)

 #image.show()

 
 embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
 embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
 ) 
 embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

 vision_model_path = "ibm-granite/granite-vision-3.2-2b"
 vision_model = Replicate(
    model=vision_model_path,
    replicate_api_token=Replicate_API_Key_Image,
    model_kwargs={
        "max_tokens": embeddings_tokenizer.max_len_single_sentence, # Set the maximum number of tokens to generate as output.
        "min_tokens": 100, # Set the minimum number of tokens to generate as output.
    },
 )
 vision_processor = AutoProcessor.from_pretrained(vision_model_path)

#Defining image prompt
 image_prompt = "If the image contains text, explain the text in the image."
 conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": image_prompt},
        ],        
    },
 ]
 vision_prompt = vision_processor.apply_chat_template(
    conversation=conversation, 
    add_generation_prompt=True,
 )
 image_Chunked_Results=""
 text = vision_model.invoke(vision_prompt, image=encode_image(image))

 return text


imagePath="static/Picture3.png"

#img=Image.open(imagePath)

#img.show()

#imagetext = get_image_text_info(img)

print("image text")
#print(imagetext)



from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

   
from flask_cors import CORS
from flask import request

app = Flask(__name__)
CORS(app)
app.jinja_env.filters['zip'] = zip

question="test"

@app.route('/')
def home():
   #return render_template('toggle.html')
   return render_template('indextest.html')
#indextest.html
print("before ask")
#response query execution and respoce function execution when ask button clicked
@app.route('/ask', methods=['POST'])
def ask():
   #question = request.form['question']
   #Pic = request.form['submit']
   #print("pic")
   #print(Pic)
   filenames=[]
   filename="Picture2.png"
   filenames.append(filename)
   filename="Picture3.png"
   filenames.append(filename)
   #audio_uris, image_uris, text_response,distances,metadatas, titles, video_uris = query_response(image_collection, question, client)
  
   print("frame ask")
   print("filename")
   print(filename)
   question="test"
   imageText=[]
   audiofiles=[]
   for file in filenames:
      text =   get_image_text_info(file)
      imageText.append(text)
      audio_file= filename.replace(".png",".mp3")
      text_to_audio(text, audio_file)
      audiofiles.append(audio_file)

   #filenames=zip(filenames,  imageText)
   print("filenames")
   print(filenames)
   print("imageText")
   print(imageText)

   return render_template('indextest.html', filenames=filenames,  imageText=imageText, question=question,audiofiles=audiofiles) 
@app.route('/ask1', methods=['POST'])
def ask1():
   #question = request.form['question']
   #Pic = request.form['imagesubmit']
   print("pic")
   #print(Pic)
   filename="Picture2.png"
   #audio_uris, image_uris, text_response,distances,metadatas, titles, video_uris = query_response(image_collection, question, client)
   image_text =   get_image_text_info(filename)
   print("frame ask1 ")
   print(image_text)
   print("filename")
   print(filename)
   
   #return render_template('popup1.html')
   #return render_template('indextest.html', image_text=image_text , filename=filename)
   return render_template('indextest.html', filenames=zip(filename,  image_text) )
  # return render_template('index.html', question=question,audio_uris=audio_uris, image_uris=image_uris,
   #                       text_response=text_response,video_uris=video_uris,frame_uris=frame_uris)
   

if __name__== '__main__':
   app.run(host="0.0.0.0", port=3025, debug=True)

#   app.run(debug=True,port=5050)

print("end program Multimodel")

