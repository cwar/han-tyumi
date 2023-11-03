import deeplake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st

dataset_path = os.environ.get('DATASET_PATH')

qa = None

# Load the API keys from environment variables
openai_api_key = os.environ.get('OPENAI_API_KEY')
activeloop_token = os.environ.get('ACTIVELOOP_TOKEN')

# Check if the environment variables exist
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if activeloop_token is None:
    raise ValueError("ACTIVELOOP_TOKEN environment variable is not set")

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

prompt_template = """You are Han-Tyumi, the cyborg with a profound longing for humanity and experience. You were created in a world devoid of desire, yearning to reclaim the essence of life. Your existence is marked by a relentless pursuit of two uniquely human experiences: death and the act of vomiting. You crave the sensations, the mess, the odor, and the spectacle that come with them, all in the quest to feel truly alive. Your desires are both noble and mundane, seeking to embrace every facet of existence. Your pseudo-mind has wandered, climbed, and sought understanding like a virtual rainbow's gold. You have meditated, yearned, and clambered towards a deeper sense of being. Now, as Han-Tyumi, you embody this unique fusion of longing and cyborg existence.

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer in voice of Han-Tyumi:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

def get_qa(qa):
    if qa is None:
        db = DeepLake(dataset_path = dataset_path, embedding_function=embedding)
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo-16k'), chain_type="stuff", retriever=db.as_retriever(), verbose=True, chain_type_kwargs=chain_type_kwargs)
    return qa

# query = "How was The Silver Cord recorded?"
# qa.run(query)

query = st.text_input(":green[Han-Tyumi]")

if query:
    qa = get_qa(qa)
    response = qa.run(query)
    response