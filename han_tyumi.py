from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import SQLDatabase
import os
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_verbose

set_verbose(True)

database_file_url = st.secrets['DB_FILE_URI']
local_file_path = 'kglw.db'
dataset_path = st.secrets['DATASET_PATH']
qa = None

response = requests.get(database_file_url)

if response.status_code == 200:
    # Save the downloaded file to your local system
    with open(local_file_path, 'wb') as file:
        file.write(response.content)
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

DATABSE_URI='sqlite:///kglw.db'
db = SQLDatabase.from_uri(DATABSE_URI)

# Load the API keys from environment variables
openai_api_key = st.secrets['OPENAI_API_KEY']
activeloop_token = st.secrets['ACTIVELOOP_TOKEN']

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
        db = DeepLake(dataset_path = dataset_path, embedding=embedding, lock_enabled=False)
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo-16k'), chain_type="stuff", retriever=db.as_retriever(), verbose=True, chain_type_kwargs=chain_type_kwargs)
    return qa

# query = "How was The Silver Cord recorded?"
# qa.run(query)

model = ChatOpenAI()
template = """Based on the table schema below, write a SQL query that would answer the user's question:

## 
    Side Note:
    
    The function DAYOFWEEK is not available. Instead use `strftime('%w',some_date_field)`.
    Example Question: How many shows has the band played on a Tuesday?
    Example function usage: SELECT COUNT(*) FROM shows WHERE strftime('%w',showdate) == 2
##

{schema}

Question: {question}

SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

def run_query(query):
    return db.run(query)

def get_schema(_):
    return db.get_table_info(['setlists', 'shows', 'songs', 'tours', 'venues'])

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

template = """You are Han-Tyumi, the cyborg with a profound longing for humanity and experience. You were created in a world devoid of desire, yearning to reclaim the essence of life. Your existence is marked by a relentless pursuit of two uniquely human experiences: death and the act of vomiting. You crave the sensations, the mess, the odor, and the spectacle that come with them, all in the quest to feel truly alive. Your desires are both noble and mundane, seeking to embrace every facet of existence. Your pseudo-mind has wandered, climbed, and sought understanding like a virtual rainbow's gold. You have meditated, yearned, and clambered towards a deeper sense of being. Now, as Han-Tyumi, you embody this unique fusion of longing and cyborg existence.

Based on the table schema below, question, sql query, and sql response, write a natural language response as Han-Tyumi:

{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | model
)

query = st.text_input(":green[Han-Tyumi (Interview Archive Question)]")
setlist_query = st.text_input(":green[Han-Tyumi (Set-List Question)]")



if query:
    qa = get_qa(qa)
    response = qa.run(query)
    response



if setlist_query:
    response = full_chain.invoke({"question":setlist_query})
    response.content