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
from langchain.globals import set_debug

set_debug(True)

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
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-4-1106-preview'), chain_type="stuff", retriever=db.as_retriever(), verbose=True, chain_type_kwargs=chain_type_kwargs)
    return qa

# query = "How was The Silver Cord recorded?"
# qa.run(query)

model = ChatOpenAI()
template = """You have access to a database for setlist data for the band King Gizzard & The Wizard Lizard.

Based on the table schema below, write a SQL query that would answer the user's question (do not answer the example questions):

## 
    Side Notes:
    
        The function DAYOFWEEK is not available. Instead use `strftime('%w',some_date_field)`.
        Example:
            Example Question: How many shows has the band played on a Tuesday?
            SQL Query: SELECT COUNT(*) FROM shows WHERE strftime('%w',showdate) = '2'

        There is no need to filter by the bands name, all the data is for one band. Avoid filtering by the bands name this may cause errors.

    Example Questions and Queries:
        Example Question: How many times has the band opened with Perihelion?
        Sql Query: select count(*) from setlists sl join songs so on sl.song_id = so.id where so.name = "perihelion" and sl.position=1

        Example Question: What was the full setlist for the show on 10-26-2022?
        Sql query:  select so.name, sl.position from shows sh join setlists sl on sh.id = sl.show_id join songs so on sl.song_id = so.id where showdate="2022-10-26" order by sl.position

        Example Question: What is the track list for the album Nonagon Infinity?
        Sql query:  select s.name, t.tracktime from tracks t join albums a on t.discography_id = a.id join songs s on s.id = t.song_id where a.albumtitle like "Nonagon Infinity" order by t.position

        Example Question: Analyze the lyrics to Petrodragonic Apocalypse for themes.
        Sql query: select s.name, s.lyrics from songs s join tracks t on t.song_id = s.id join albums a on t.discography_id =  a.id where a.albumtitle like "%Petrodragonic%" order by t.position

        Example Question: What are all the songs on the albums Petrodragonic Apocalypse and The Silver Cord?
        Sql Query: select a.albumtitle, t.position, s.name from songs s join tracks t on t.song_id = s.id  join albums a on t.discography_id =  a.id  where a.albumtitle like "%Petrodragonic%" or a.albumtitle = "The Silver Cord" order by a.id, t.position
##



{schema}

Question: {question}

SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

def run_query(query):
    return db.run(query)

def get_schema(_):
    return db.get_table_info(['setlists', 'shows', 'albums', 'songs', 'tracks', 'tours', 'venues'])

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
    | ChatOpenAI(model='gpt-4-1106-preview')
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