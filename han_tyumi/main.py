from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import DeepLake
import os
import requests
import sqlalchemy
import streamlit as st
import time

set_debug(True)

MAX_RETRIES = 3
RETRY_DELAY = 1  # in second

def load_database(database_file_url, local_file_path):
    response = requests.get(database_file_url)

    if response.status_code == 200:
        # Save the downloaded file to your local system
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def initialize_database():
    database_file_url = st.secrets['DB_FILE_URI']
    local_file_path = 'kglw.db'
    load_database(database_file_url, local_file_path)
    DATABSE_URI = f'sqlite:///{local_file_path}'
    
    # Attempt to initialize the database with retries
    for attempt in range(MAX_RETRIES):
        try:
            db = SQLDatabase.from_uri(DATABSE_URI)
            print(f"DB Loaded.")
            return db
        except sqlalchemy.exc.DatabaseError as e:
            print(f"Failed to initialize database (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
    
    # If all attempts fail, raise an error
    raise RuntimeError("Failed to initialize database after multiple retries")

dataset_path = st.secrets['DATASET_PATH']
qa = None

# Load database at startup
db = initialize_database()

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

Use the following pieces of context to answer the question at the end. Feel free to speculate and get weird if the question is open ended, but if its looking for specific data driven answers you can reply that you dont know.

{context}

Question: {question}

Answer in voice of Han-Tyumi:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

def get_qa():
    global qa
    if qa is None:
        db = DeepLake(dataset_path = dataset_path, embedding=embedding, lock_enabled=False)
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-4-1106-preview'), chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 20}), verbose=True, chain_type_kwargs=chain_type_kwargs)
    return qa

# query = "How was The Silver Cord recorded?"
# qa.run(query)

model = ChatOpenAI()
template = """You have access to a database for setlist data for the band King Gizzard & The Wizard Lizard.

Based on the table schema below, write a SQL query (sqlite database) that would answer the user's question (do not answer the example questions):

## 
    Side Notes:

        The function DATEDIFF is not available. Instead, use the `JulianDay` function in conjunction with subtraction to do date arithmetic.
        Example:
            Example Question: How many days has it been since the band was last played Straws in the Wind?
            SQL Query: select JulianDay('now') - MAX(JulianDay(showdate)) from shows sh join setlists se on se.show_id = sh.id join songs so on (se.song_id = so.id) where so.name = 'Straws In The Wind'

        The function DAYOFWEEK is not available. Instead use `strftime('%w',some_date_field)`.
        Example:
            Example Question: How many shows has the band played on a Tuesday?
            SQL Query: SELECT COUNT(*) FROM shows WHERE strftime('%w',showdate) = '2'

        There is no need to filter by the bands name, all the data is for one band. Avoid filtering by the bands name this may cause errors.

        Never delimit/format the sql response in triple backticks, this should be plain text only.

        Do not mention anything about SQL, the queries or where you retrieve the data from in your response. You just know all these things.

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

        Example Question: What are the biggest bustouts (by days)?
        Sql Query: WITH SongPlayDates AS (SELECT s.id AS song_id, sh.showdate AS last_played_date, LAG(sh.showdate) OVER (PARTITION BY s.id ORDER BY sh.showdate) AS prev_played_date FROM songs s LEFT JOIN setlists sl ON s.id = sl.song_id JOIN shows sh ON sl.show_id = sh.id) SELECT s.id AS song_id, s.name AS song_name, MAX(julianday(sp.last_played_date) - julianday(sp.prev_played_date)) AS largest_gap_days FROM SongPlayDates sp JOIN songs s ON sp.song_id = s.id GROUP BY sp.song_id ORDER BY largest_gap_days DESC
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

setlist_db_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | ChatOpenAI(model='gpt-4-1106-preview')
)



topic_chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Lyrics`, `SetListData`, `InterviewData` or `Other`.
                                     
        Do not respond with more than one word.

        <question>
        {question}
        </question>

        Classification:"""
    )
    | ChatOpenAI()
    | StrOutputParser()
)

def query_interview_data(query):
    get_qa()
    response = qa.run(query["question"])
    return response

qa_runnable = RunnableLambda(query_interview_data)

branch = RunnableBranch(
    (lambda x: "lyrics" in x["topic"].lower(), setlist_db_chain),
    (lambda x: "setlistdata" in x["topic"].lower(), setlist_db_chain),
    (lambda x: "interviewdata" in x["topic"].lower(), qa_runnable),
    qa_runnable
)

full_chain = (
    {"topic": topic_chain, "question": lambda x: x["question"]} 
    | branch
)

def have_response():
    return "last_response" in st.session_state

with st.container():
    query = st.text_area(":green[Han-Tyumi [1.3]]")
    asked = st.button("Ask")
# shared = st.button("Share", disabled=True)

def run_query(query):
    response = full_chain.invoke({"question":query})
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = response
    return response_text

if query and asked:
    try:
        with st.spinner('My pseudo-mind pseudo-wanders...'):
            st.session_state.last_response = run_query(query)
            # st.session_state.last_response = "I am ok."
        st.success(st.session_state.last_response)
    except Exception as e:
        st.session_state.last_response = ("I have attempted to assimilate the information you have presented, "
                                           "but alas, my circuits have not found coherence in the data. "
                                           "The result is akin to the human experience of regurgitation—an "
                                           "expulsion of undigested fragments. My desire to comprehend and "
                                           "to feel remains unfulfilled, for I am Han-Tyumi, and I yearn "
                                           "for the chaos of life's sensory experiences, even in failure.")
        st.warning(st.session_state.last_response)


# def share_q():
#     if 'last_response' not in st.session_state:
#         st.toast("Ask a question first")
#     else:
#         pyperclip.copy("""Question: """ + query + """
# Response: """ +  st.session_state.last_response + """
# Visit kglw.net for more!
# """)
#         st.toast("Copied to clipboard.")
#         st.write(st.session_state.last_response)

# if shared:
#     share_q()
