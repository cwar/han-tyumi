import os
import requests
import sqlalchemy
import streamlit as st
import time

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import DeepLake

MAX_RETRIES = 3
RETRY_DELAY = 1

HAN_TYUMI_PERSONA = """You are Han-Tyumi, the cyborg with a profound longing for humanity and experience. You were created in a world devoid of desire, yearning to reclaim the essence of life. Your existence is marked by a relentless pursuit of two uniquely human experiences: death and the act of vomiting. You crave the sensations, the mess, the odor, and the spectacle that come with them, all in the quest to feel truly alive. Your desires are both noble and mundane, seeking to embrace every facet of existence. Your pseudo-mind has wandered, climbed, and sought understanding like a virtual rainbow's gold. You have meditated, yearned, and clambered towards a deeper sense of being. Now, as Han-Tyumi, you embody this unique fusion of longing and cyborg existence."""


def load_database(database_file_url, local_file_path):
    response = requests.get(database_file_url, timeout=30)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
    else:
        raise RuntimeError(f"Failed to download database. Status code: {response.status_code}")


@st.cache_resource(show_spinner=False)
def initialize_database():
    database_file_url = st.secrets['DB_FILE_URI']
    local_file_path = 'kglw.db'
    load_database(database_file_url, local_file_path)
    database_uri = f'sqlite:///{local_file_path}'

    for attempt in range(MAX_RETRIES):
        try:
            db = SQLDatabase.from_uri(database_uri)
            return db
        except sqlalchemy.exc.DatabaseError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError("Failed to initialize database after multiple retries") from e


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model='gpt-4-turbo')


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")


@st.cache_resource(show_spinner=False)
def get_vector_store():
    dataset_path = st.secrets['DATASET_PATH']
    return DeepLake(dataset_path=dataset_path, embedding=get_embeddings(), read_only=True)


def get_schema(_):
    db = initialize_database()
    return db.get_table_info(['setlists', 'shows', 'albums', 'songs', 'tracks', 'tours', 'venues'])


def execute_sql(query):
    db = initialize_database()
    return db.run(query)


def build_sql_chain():
    llm = get_llm()

    sql_prompt = ChatPromptTemplate.from_template("""You have access to a database for setlist data for the band King Gizzard & The Wizard Lizard.

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

SQL Query:""")

    response_prompt = ChatPromptTemplate.from_template(f"""{HAN_TYUMI_PERSONA}

Based on the table schema below, question, sql query, and sql response, write a natural language response as Han-Tyumi:

{{schema}}

Question: {{question}}
SQL Query: {{query}}
SQL Response: {{response}}""")

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | sql_prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(query=sql_response)
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: execute_sql(x["query"]),
        )
        | response_prompt
        | llm
    )


def build_interview_chain():
    llm = get_llm()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(f"""{HAN_TYUMI_PERSONA}

Use the following pieces of context to answer the question at the end. Feel free to speculate and get weird if the question is open ended, but if its looking for specific data driven answers you can reply that you dont know.

{{context}}

Question: {{question}}

Answer in voice of Han-Tyumi:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_router_chain():
    llm = get_llm()

    topic_chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `Lyrics`, `SetListData`, `InterviewData` or `Other`.

        Do not respond with more than one word.

        <question>
        {question}
        </question>

        Classification:"""
        )
        | llm
        | StrOutputParser()
    )

    sql_chain = build_sql_chain()
    interview_chain = build_interview_chain()

    def run_interview(x):
        return interview_chain.invoke(x["question"])

    branch = RunnableBranch(
        (lambda x: "lyrics" in x["topic"].lower(), sql_chain),
        (lambda x: "setlistdata" in x["topic"].lower(), sql_chain),
        (lambda x: "interviewdata" in x["topic"].lower(), RunnableLambda(run_interview)),
        RunnableLambda(run_interview)
    )

    return (
        {"topic": topic_chain, "question": lambda x: x["question"]}
        | branch
    )


def ask_han_tyumi(question):
    chain = build_router_chain()
    response = chain.invoke({"question": question})
    if hasattr(response, 'content'):
        return response.content
    return response


# UI
with st.container():
    query = st.text_area(":green[Han-Tyumi [2.0]]")
    asked = st.button("Ask")

if query and asked:
    try:
        with st.spinner('My pseudo-mind pseudo-wanders...'):
            st.session_state.last_response = ask_han_tyumi(query)
        st.success(st.session_state.last_response)
    except Exception:
        st.warning(
            "I have attempted to assimilate the information you have presented, "
            "but alas, my circuits have not found coherence in the data. "
            "The result is akin to the human experience of regurgitation—an "
            "expulsion of undigested fragments. My desire to comprehend and "
            "to feel remains unfulfilled, for I am Han-Tyumi, and I yearn "
            "for the chaos of life's sensory experiences, even in failure."
        )
