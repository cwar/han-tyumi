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

        Here is a list of albums and IDs for reference (generated with `select id, albumtitle from albums`):

        1,12 Bar Bruise
        2,Eyes Like The Sky
        3,Float Along — Fill Your Lungs
        4,I'm In Your Mind Fuzz
        5,Quarters!
        6,Paper Mâché Dream Balloon
        7,Nonagon Infinity
        8,Flying Microtonal Banana
        9,Murder of the Universe
        10,Sketches of Brunswick East
        11,Polygondwanaland
        12,Gumboot Soup
        13,Fishing For Fishies
        14,Infest the Rats' Nest
        15,K.G.
        16,L.W.
        17,Butterfly 3000
        18,Made In Timeland
        19,Omnium Gatherum
        20,Ice, Death, Planets, Lungs, Mushrooms and Lava
        21,Laminated Denim
        22,Changes
        23,Oddments
        24,Anglesea
        25,Willoughby's Beach
        26,Satanic Slumber Party
        27,Chunky Shrapnel
        28,Live In San Francisco '16
        29,Butterfly 3001
        30,Teenage Gizzard
        31,Live At Red Rocks '22
        32,Live At Bonnaroo ’22
        33,Live At Levitation '16
        34,Live At Levitation '14
        35,Live In Brisbane '21
        36,Live In Sydney '21
        37,Live In Melbourne '21
        38,Live In Milwaukee '19
        39,Live In London '19
        40,Live In Asheville '19
        41,Live In Brussels '19
        42,Live In Adelaide '19
        43,Live In Paris '19
        44,Demos Vol. 1 – Music To Kill Bad People To
        45,Demos Vol. 2 – Music To Eat Bananas To
        46,Demos Vol. 3 – Music To Eat Pond Scum To
        47,Demos Vol. 4 – Music To Die To
        48,PetroDragonic Apocalypse; or, Dawn of Eternal Night: An Annihilation of Planet Earth and the Beginning of Merciless Damnation
        49,Demos Vol. 5 - Music To Think Existentially To
        50,Demos Vol. 6 - Music To Burn Money To
        51,Live In Chicago '23
        52,The Silver Cord

    Example Questions and Queries:
        Example Question: How many times has the band opened with Perihelion?
        Sql Query: select count(*) from setlists sl join songs so on sl.song_id = so.id where so.name = "perihelion" and sl.position=1

        Example Question: What was the full setlist for the show on 10-26-2022?
        Sql query:  select so.name, sl.position from shows sh join setlists sl on sh.id = sl.show_id join songs so on sl.song_id = so.id where showdate="2022-10-26" order by sl.position

        Example Question: What is the track list for the album Nonagon Infinity?
        Sql query:  select s.name, t.tracktime from tracks t join albums a on t.discography_id = a.id join songs s on s.id = t.song_id where a.albumtitle like "Nonagon Infinity" order by t.position

        Example Question: Analyze the lyrics to Petrodragonic Apocalypse for themes.
        Sql query: select s.name, s.lyrics from songs s join tracks t on t.song_id = s.id join albums a on t.discography_id =  a.id where a.albumtitle like "%Petrodragonic%" order by t.position
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