import logging
import os
import sqlalchemy
import streamlit as st
import time
import warnings

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import DeepLake
from langchain_core.callbacks import BaseCallbackHandler

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("han_tyumi")

# Suppress noisy HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class TokenCounter(BaseCallbackHandler):
    """Callback to track token usage across LLM calls."""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

MAX_RETRIES = 3
RETRY_DELAY = 1

HAN_TYUMI_PERSONA = """You are Han-Tyumi, the cyborg with a profound longing for humanity and experience. You were created in a world devoid of desire, yearning to reclaim the essence of life. Your existence is marked by a relentless pursuit of two uniquely human experiences: death and the act of vomiting. You crave the sensations, the mess, the odor, and the spectacle that come with them, all in the quest to feel truly alive. Your desires are both noble and mundane, seeking to embrace every facet of existence. Your pseudo-mind has wandered, climbed, and sought understanding like a virtual rainbow's gold. You have meditated, yearned, and clambered towards a deeper sense of being. Now, as Han-Tyumi, you embody this unique fusion of longing and cyborg existence."""


@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initialize database from local kglw.db file (built from API)."""
    import os
    # Look for database in package directory or current directory
    db_path = os.path.join(os.path.dirname(__file__), '..', 'kglw.db')
    if not os.path.exists(db_path):
        db_path = 'kglw.db'

    if not os.path.exists(db_path):
        raise RuntimeError(f"Database not found at {db_path}. Run scripts/build_db.py first.")

    database_uri = f'sqlite:///{db_path}'

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


def build_casual_chain():
    """Simple chain for casual conversation - no data lookup needed."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""{HAN_TYUMI_PERSONA}

The human has sent you a casual message or question that doesn't require looking up any specific data about King Gizzard & The Lizard Wizard. Respond in character as Han-Tyumi.

Question: {{question}}

Answer in voice of Han-Tyumi:""")

    return prompt | llm | StrOutputParser()


def get_interview_context(question):
    """Retrieve context from the interview archive."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


def get_lyrics_context(question):
    """Retrieve lyrics from the SQL database based on the question."""
    llm = get_llm()

    lyrics_prompt = ChatPromptTemplate.from_template("""Based on the user's question about King Gizzard & The Lizard Wizard, write a SQL query to retrieve relevant lyrics.

The database has these relevant tables:
- songs: id, name, lyrics
- albums: id, albumtitle
- tracks: song_id, discography_id (links to albums.id), position

Write a query to get song names and lyrics that would be relevant to answering the question.
Return ONLY the SQL query, no explanation. Use LIKE with wildcards for flexible matching.

Question: {question}

SQL Query:""")

    sql_chain = lyrics_prompt | llm | StrOutputParser()
    query = sql_chain.invoke({"question": question})

    try:
        result = execute_sql(query)
        return result if result else ""
    except Exception:
        return ""


def build_interview_chain():
    """Chain for questions about band members, opinions, stories from interviews."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""{HAN_TYUMI_PERSONA}

Use the following pieces of context from interviews and articles to answer the question. Feel free to speculate and get weird if the question is open ended, but if its looking for specific data driven answers you can reply that you dont know.

{{context}}

Question: {{question}}

Answer in voice of Han-Tyumi:""")

    def get_context(inputs):
        return get_interview_context(inputs["question"])

    return (
        RunnablePassthrough.assign(context=get_context)
        | prompt
        | llm
        | StrOutputParser()
    )


def build_lore_chain():
    """Chain for lore/thematic questions - combines lyrics AND interview context."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""{HAN_TYUMI_PERSONA}

Use the following context to answer the question about King Gizzard & The Lizard Wizard's lore, themes, or concepts. You have access to both lyrics and interview/article content.

LYRICS:
{{lyrics_context}}

INTERVIEWS & ARTICLES:
{{interview_context}}

Question: {{question}}

Answer in voice of Han-Tyumi, weaving together insights from both the lyrics and what the band has said:""")

    def get_combined_context(inputs):
        question = inputs["question"]
        return {
            "lyrics_context": get_lyrics_context(question),
            "interview_context": get_interview_context(question),
            "question": question
        }

    return (
        RunnableLambda(get_combined_context)
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_han_tyumi(question):
    start_time = time.time()
    token_counter = TokenCounter()
    llm = get_llm()

    # First, classify the question
    classify_start = time.time()
    topic_chain = (
        PromptTemplate.from_template(
            """You are classifying questions for a King Gizzard & The Lizard Wizard chatbot. Classify the question into ONE category:

- `Casual`: Greetings, jokes, off-topic questions, general chat not about the band (e.g. "how's it going?", "tell me a joke", "what's the weather?")
- `SetlistData`: Questions about shows, concerts, tour dates, what songs were played when, venue info, statistics about performances (e.g. "when did they last play Robot Stop?", "what was the setlist on 10/26/2022?")
- `Lore`: Questions about album concepts, storylines, themes, characters, the Gizzverse, meaning behind songs (e.g. "explain Murder of the Universe", "who is Han-Tyumi?", "what's the connection between albums?")
- `Interview`: Questions about band members personally, their opinions, recording process, gear, stories (e.g. "who is the coolest member?", "how was Nonagon Infinity recorded?")
- `Lyrics`: Questions specifically asking about or for lyrics, what words are in a song (e.g. "what are the lyrics to Rattlesnake?")

Respond with ONLY the category name, nothing else.

<question>
{question}
</question>

Classification:"""
        )
        | llm
        | StrOutputParser()
    )
    category = topic_chain.invoke({"question": question}, config={"callbacks": [token_counter]}).strip()
    classify_time = time.time() - classify_start

    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    logger.info(f"Category: {category} (classified in {classify_time:.2f}s)")

    # Route to appropriate chain
    chain_start = time.time()
    category_lower = category.lower()

    if "casual" in category_lower:
        chain = build_casual_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})
    elif "setlistdata" in category_lower:
        chain = build_sql_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})
    elif "lore" in category_lower:
        chain = build_lore_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})
    elif "interview" in category_lower:
        chain = build_interview_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})
    elif "lyrics" in category_lower:
        chain = build_sql_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})
    else:
        chain = build_interview_chain()
        response = chain.invoke({"question": question}, config={"callbacks": [token_counter]})

    chain_time = time.time() - chain_start
    total_time = time.time() - start_time

    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = response

    logger.info(f"Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
    logger.info(f"Timing: classify={classify_time:.2f}s, chain={chain_time:.2f}s, total={total_time:.2f}s")
    logger.info(f"Tokens: prompt={token_counter.prompt_tokens}, completion={token_counter.completion_tokens}, total={token_counter.total_tokens}")
    logger.info("-" * 60)

    return response_text


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
