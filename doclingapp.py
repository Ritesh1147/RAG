import sys
assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12 to run this notebook."
#print("test")
import platform
#print(platform.python_version())

import os
import logging
logging.basicConfig(level=logging.INFO)

#Replicate_API_Key = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"
Replicate_API_Key = "r8_eEeSN9JpBRARoo5l4v0DrcDJ3CwqWji3KryIL"

#os.environ["REPLICATE_API_TOKEN"] = "r8_5XD4d13F1lo1DuKmYnmH4g7raFwmrIL0Sl0pX"

os.environ["REPLICATE_API_TOKEN"] =  "r8_eEeSN9JpBRARoo5l4v0DrcDJ3CwqWji3KryIL"

import base64
import io
import PIL.Image
import PIL.ImageOps
from IPython.display import display



# Get the chart no related to the response text if there is chart present is the answer
def get_chart_no_from_text(responsestr ):
   responseChartNos =""
   chartPos=responsestr.find("Chart ")
   print("Chartpos")
   print(chartPos)
   rightStr=responsestr
   while chartPos > 0:
      rightStr = rightStr[chartPos + 6:]
      print(rightStr)
      chartNopos = rightStr.find(" ")
      if chartNopos <=0:
        ChartNoStr=rightStr
      else:   
       ChartNoStr = rightStr[:chartNopos]
       print("chatNostr")
       print(ChartNoStr)
      if responseChartNos == "":
       responseChartNos =  ChartNoStr
      else:
       responseChartNos = responseChartNos +", " + ChartNoStr 
       
       if chartNopos > 0:
        rightStr= rightStr[chartNopos + 1:]
       else:
        rightStr=""
      chartPos = rightStr.find("Chart ")
      print("Chart Nos")
      print(responseChartNos)
      
      return responseChartNos
   

         
# encode image used in converting the image to text
def encode_image(image: PIL.Image.Image, format: str = "png") -> str:
    image = PIL.ImageOps.exif_transpose(image) or image
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format)
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uri = f"data:image/{format};base64,{encoding}"
    return uri

# to retrieve image and text for the query
from llama_index.core.response.notebook_utils import display_source_node,display_image,  display_image_uris , ImageNode
def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings 
from transformers import AutoTokenizer

from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
from transformers import AutoProcessor

import itertools
from docling_core.types.doc.document import RefItem

from langchain_core.documents import Document

def get_image_text(image):
 
 embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
 embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
 )
 embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

 vision_model_path = "ibm-granite/granite-vision-3.2-2b"
 vision_model = Replicate(
    model=vision_model_path,
    replicate_api_token=Replicate_API_Key,
    model_kwargs={
        "max_tokens": embeddings_tokenizer.max_len_single_sentence, # Set the maximum number of tokens to generate as output.
        "min_tokens": 100, # Set the minimum number of tokens to generate as output.

        # Change max_tokens to 1000 and min_tokens to 200
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
        


#defining the embed model for text and vision model 
def create_vectordb():
 embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
 embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
 )
 embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)


 import time

 print("Program started")
 time.sleep(20)
 print("Program finished after 20 seconds")


 vision_model_path = "ibm-granite/granite-vision-3.2-2b"
 vision_model = Replicate(
    model=vision_model_path,
    replicate_api_token=Replicate_API_Key,
    model_kwargs={
        "max_tokens": embeddings_tokenizer.max_len_single_sentence, # Set the maximum number of tokens to generate as output.
        "min_tokens": 100, # Set the minimum number of tokens to generate as output.
    },
 )
 vision_processor = AutoProcessor.from_pretrained(vision_model_path)




 from docling.document_converter import DocumentConverter, PdfFormatOption
 from docling.datamodel.base_models import InputFormat
 from docling.datamodel.pipeline_options import PdfPipelineOptions

#pdf pipeline options
 pdf_pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    generate_picture_images=True,
 )
 format_options = {
    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
 }


# Defining the source document
 converter = DocumentConverter(format_options=format_options)
 #https://boj.org.jm/wp-content/uploads/2025/03/ABM-Performance-Table-January-2025-Region-Parish.pdf
 sources = [
    "http://0.0.0.0:4060/dsa_pdf.pdf",
 #   "https://boj.org.jm/wp-content/uploads/2024/10/BOJ-Macroprudential-Policy-Report-June-24.pdf",
 ]#/pictures/80
 conversions = { source: converter.convert(source=source).document for source in sources }


 from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
 from docling_core.types.doc.document import TableItem
 #from langchain_core.documents import Document

 #creating the chunks for the document
 doc_id = 0
 texts: list[Document] = []
 for source, docling_document in conversions.items():
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(docling_document):
        items = chunk.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue # we will process tables later
        refs = " ".join(map(lambda item: item.get_ref().cref, items))
 #        print(refs)
        text = chunk.text
        document = Document(
            page_content=text,
            metadata={
                "doc_id": (doc_id:=doc_id+1),
                "source": source,
                "ref": refs,
            },
        )
        texts.append(document)

#print(f"{len(texts)} text document chunks created")

 from docling_core.types.doc.labels import DocItemLabel

 # adding the tables in the document 
 doc_id = len(texts)
 tables: list[Document] = []
 for source, docling_document in conversions.items():
    for table in docling_document.tables:
        if table.label in [DocItemLabel.TABLE]:
            ref = table.get_ref().cref
 #            print(ref)
            text = table.export_to_markdown()
            document = Document(
                page_content=text,
                metadata={
                    "doc_id": (doc_id:=doc_id+1),
                    "source": source,
                    "ref": ref
                },
            )
            tables.append(document)


 #print(f"{len(tables)} table documents created")


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
 pictures: list[Document] = []
 doc_id = len(texts) + len(tables)
 for source, docling_document in conversions.items():
    for picture in docling_document.pictures:
        time.sleep(6)
        ref = picture.get_ref().cref
        print(ref)
        image = picture.get_image(docling_document)
        if image:
            text = vision_model.invoke(vision_prompt, image=encode_image(image))
            document = Document(
                page_content=text,
                metadata={
                    "doc_id": (doc_id:=doc_id+1),
                    "source": source,
                    "ref": ref,
                },
            )
        
            print("image_text")
            print(text)
            image_Chunked_Results = image_Chunked_Results + text + "\n"
            print("image doc id")
            print(doc_id)
            pictures.append(document)

# print(f"{len(pictures)} image descriptions created")

#image_Chunked_Results
# image_chunk_data_file_name="/home/mkandra/Documents/BOJ/BOJMultiModelRAG/chunkTextdata/image_Chunked_Results.txt"
 image_chunk_data_file_name="chunkTextdata/image_Chunked_Results.txt"
 with open(image_chunk_data_file_name, "w") as f:
  f.write(image_Chunked_Results)

# Print all pictures in the created documents
 image_no=0
 for document in pictures:
  print("print document")
  print(document)
 # print(f"Document ID: {document.metadata['doc_id']}")
  source = document.metadata['source']
  doc_id = document.metadata['doc_id']
# print(f"Source: {source}")
# print(f"Content:\n{document.page_content}")
  docling_document = conversions[source]
  ref = document.metadata['ref']
  picture = RefItem(cref=ref).resolve(docling_document)
  image = picture.get_image(docling_document)
  image_no = image_no + 1
  image_path= "InputImages/image" + str(image_no) + ".jpeg"
  image.save(image_path)
  print("Image:")
  print(ref)
  print("doc_id")
  print(doc_id)


#    print(document)
 #   image.show()
 #   display(image)
  #  print("=" * 80) # Separator for clarity
    
 #creating vector store database and adding texts, tables and images to vector store   
 import tempfile
 from langchain_core.vectorstores import VectorStore
 from langchain_milvus import Milvus
 '''
 db_file = tempfile.NamedTemporaryFile(prefix="vectorstore_", suffix=".db", delete=False).name
 #print(f"The vector database will be saved to {db_file}")

 vector_db: VectorStore = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    enable_dynamic_field=True,
    index_params={"index_type": "AUTOINDEX"},
 )
 documents = list(itertools.chain(texts, tables, pictures))
 ids = vector_db.add_documents(documents)
 print("documents added to vectordb")
 answer=""


 return vector_db,documents, ids,  pictures ,conversions
 '''

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
 documents = list(itertools.chain(texts, tables, pictures))
 ids = vector_store.add_documents(documents)
 print("documents added to vectordb")
 answer=""

 #return documents
 return vector_store,documents, ids,  pictures ,conversions



 from langchain.prompts import PromptTemplate
 from langchain.chains.retrieval import create_retrieval_chain
 from langchain.chains.combine_documents import create_stuff_documents_chain

#querying the documents from the retriever for the query
def query_vectordb(vector_db,documents, ids,  query, pictures):
 
 docs=vector_db.as_retriever().invoke(query)

 
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
 chartNos = get_chart_no_from_text(answer)
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
 ''' for file_instatic in os.listdir('/home/mkandra/Documents/BOJ/BOJMultiModelRAG/static/'):
  if os.path.isfile(file_instatic):
    os.remove(file_instatic)
 '''
 for retrieveDoc in retrievedDocs:
#   print(i)
#   print("retrived doc " )
#   print(retrievedDocs[i])
#   print("retrived doc  metadata")
#    print(retrievedDocs[i].metadata)
#   print("retrived doc  metadata doc id")
#   print(retrievedDocs[i].metadata['doc_id'])
   answer_docid=retrievedDocs[i].metadata['doc_id']
   source=retrievedDocs[i].metadata['source']
   response_pagecontent=retrievedDocs[i].page_content
   tupleItem=(answer_docid,source,response_pagecontent)
   tupleResponse.append(tupleItem)
#    answer_docid=52
#   print("pictures")
#    print(pictures)
   i += 1
   match,responsePic = get_picture(pictures,page_content,answer_docid,conversions)
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
 return answer,page_content,source,chartNos, pictures,outputs, responsePictures,tupleResponse

def get_picture (pictures,docling_document,answer_docid,conversions):
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
  match="No"
  for pic in pictures:
    print("for Pic")
    print(pic)
#   print(pic.metadata['doc_id'])
#   print(answer_docid)
    if pic.metadata['doc_id']==answer_docid:
     print("match")
     match="Yes"
     source = pic.metadata['source']
     pic_document = conversions[source]
     print("pic document get picture")
#     print(pic_document)
     ref = pic.metadata['ref']
     responsepicture = RefItem(cref=ref).resolve(pic_document)
     responsePic = responsepicture.get_image(pic_document)
  if match=="No":
      responsePic=None
  return match, responsePic


from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from flask import request
from docling_core.types.doc.document import PictureDescriptionData
from IPython import display


#configuring env path for langsmith
#env_path = r'/opt/thelvwapp/Documents/MultiModelRAG/doclingver1.1/.env'
env_path = r'.env'
load_dotenv(env_path)


vector_db,documents, ids,  picture,conversions = create_vectordb()
###Falsk script to display input box and response in html page

#defning the flask app
''''
Flask Commen start

app = Flask(__name__)
CORS(app)
import itertools
from langchain_core.documents import Document

print("before route")
@app.route('/')
def home():
   return render_template('index.html')
   question = request.form['question']
   answer,page_content,source, chartNos,/opt/thelvwapp/Documents/MultiModelRAG/doclingver1.1pictures,docling_document,responsePictures,tupleResponse = query_vectordb(vector_db,documents, ids,  question, picture)
   print(answer)
   print(responsePictures)
   return render_template('index.html', question=question,answer=answer,page_content=page_content,source=source,chartNos=chartNos,file_names=responsePictures,
                          tupleResponse=tupleResponse)
   
if __name__== '__main__':
   app.run(host="0.0.0.0", port=7003, debug=True)
#   app.run(debug=True,port=5050)
Fllask Comment end
'''