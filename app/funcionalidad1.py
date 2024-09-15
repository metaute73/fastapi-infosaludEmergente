import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import json

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
#Deserializamos el vector store
faiss_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = faiss_store.as_retriever()
parser = StrOutputParser()
#Deserializamos nuestra prompt template
with open('prompt_template.json', 'r') as f:
    prompt_dict = json.load(f)
prompt = PromptTemplate.from_template(template=prompt_dict["template"])

#Deserializamos nuestra configuración del modelo
with open('model_config.json', 'r') as f:
    model_config = json.load(f)
model = ChatOpenAI(model_name=model_config["model_name"], temperature=model_config["temperature"])


result = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = result | prompt | model | parser

def usar_infoSE(question):
  return chain.invoke(question)

