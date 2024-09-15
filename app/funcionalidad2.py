from langchain.document_loaders import YoutubeLoader
from langchain.chains import LLMChain
from langchain.llms import OpenAI as openai
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI as openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import torch

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_url(video_url: str)-> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript =loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db, query, k=4):
    #text-davinci can handle 4097 tokens
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm =openai(model="gpt-3.5-turbo-instruct")
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template= """

          Eres un asistente inteligente de videos en Youtube

          Responde la siguiente pregunta: {question}
          Buscando en el siguiente guion de youtube: {docs}

          Solo responde la pregunta ayudandote del guion de youtube.

          Si crees no tener suficiente información para responder la pregunta, responde "No tengo suficiente información para responder la pregunta"
          No utilices otra información para responder la pregunta
          Responde en español
          Tu respuesta debería estar detallada.
        """
    )
    chain = LLMChain(llm=llm, prompt = prompt)
    response = chain.run(question=query,docs=docs_page_content)
    response = response.replace("\n", "")
    return response

def usar_infoSE_2(video, pregunta):
  '''
  Tips para la prevención y control del cólera: https://www.youtube.com/watch?v=g8UnFD3lmzk

  Cómo empezó: https://www.youtube.com/watch?v=k8J6wEsKDSk


  Etapas del VIH: https://www.youtube.com/watch?v=xmk_ZrKDwJs

  La plaga de nuestro tiempo: https://www.youtube.com/watch?v=hgcQfIt1kqw


  Pandemia del COVID-19 (Documental): https://www.youtube.com/watch?v=4DJtjyB1gvE

  ¿Cómo funcionan las vacunas mRNA?: https://www.youtube.com/watch?v=Be4GLTiawrQ
  '''
  url = ''
  if video == 1:
    url = 'https://www.youtube.com/watch?v=g8UnFD3lmzk'
  elif video == 2:
    url = 'https://www.youtube.com/watch?v=k8J6wEsKDSk'
  elif video == 3:
    url = 'https://www.youtube.com/watch?v=xmk_ZrKDwJs'
  elif video == 4:
    url = 'https://www.youtube.com/watch?v=hgcQfIt1kqw'
  elif video == 5:
    url = 'https://www.youtube.com/watch?v=4DJtjyB1gvE'
  else:
    url = 'https://www.youtube.com/watch?v=Be4GLTiawrQ'

  db = create_vector_db_from_youtube_url(url)
  response = get_response_from_query(db, pregunta)
  return response

#print(usar_infoSE_2(1, '¿what is the video about?'))