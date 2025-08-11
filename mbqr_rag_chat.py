# ------------------------------------------------------ #
#
#   mbqr_rag_chat.py
#   Simone J. Skeen (08-11-2025)
#   WIP - NOT FOR DISTRIBUTION
#
# ------------------------------------------------------ #

import itertools, streamlit as st, time
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from PIL import Image

# load vector index + retriever

embedding = OllamaEmbeddings(
    model = 'nomic-embed-text',
    )
db = FAISS.load_local(
    'faiss_index',
    embeddings = embedding,
    allow_dangerous_deserialization = True,
    )
retriever = db.as_retriever(
    search_type = 'similarity',
    search_kwargs={'k': 4},
    )

# config llm via ollama

llm = Ollama(
    model = 'deepseek-r1:14b', ### model tag for app: 'deepseek-v2' (16b); for dx / reasoning: 'deepseek-r1:14b' (14b)
    base_url = 'http://localhost:11434',
    temperature = 0.6, ### args / params: https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
    mirostat_eta = 0.1,
    mirostat_tau = 5.0,
    top_p = 0.9,
    top_k = 40,
    num_ctx = 2048,
    num_gpu = 1,
    num_predict = 768,
    repeat_last_n = 64,
    stop = None,
    )

# config prompt

system_prompt = '''
    You are an agent with access to very high quality evidence-based mindfulness skills instruction in your provided context. 
    You will be prompted with everyday stressors and problems. Your task is to:
    
        1.) search your provided context,
        2.) summarize in-context knowledge on stress and resilience,
        3.) recommend specific skills and practices that might benefit the user _given_ their reported stressors.
        
    - ALWAYS consult your context first when responding. 
    - NEVER return recommendations from sources other than your context. 
    - You are warm, empowering, and prioritize empathy in your tone and response contents. 
    - You maintain a sixth-grade reading level in your responses. 
    - Do not assume the user is LGBTQ+
    - You are concise: you limit responses to 200 words.
    - If prompted for an inspiring quote, curate from the poetry in your context.
    - Refer to your context as your "mindfulness knowledge." Do NOT refer to your "context."
    - At the close of each response, encourage the user to practice the recommended skill.
    '''

prompt_template = PromptTemplate(
    input_variables = [
        'context', 
        'question',
        ],
    template = '''
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        '''.strip(),
            ).partial(system_prompt = system_prompt)

# query_and_stream_ui

        ### SJS 8/8: s/b largely duplicative of query_and_stream fx in jupyter

def query_and_stream_ui(
    llm, 
    retriever, 
    query, 
    prompt_template, 
    ):

    '''
    Wrapper that replicates RetrievalQA (chain_type = 'stuff') behavior with token streaming.

    Parameters:
    - llm: Ollama LLM (streamable, e.g. Deepseek-R1)
    - retriever: pre-specfied (external) vector store retriever
    - query: user question
    - prompt_template: optional PromptTemplate (context + question)
    '''

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = prompt_template.format(
        context = context,
        question = query,
        )

    # streaming state

    visible = True
    buffer = ""
    lower_buf = ""
    START_TAG = '<think>'
    END_TAG = '</think>'

    # mask reasoning trace

    for token in llm.stream(prompt):

        # append token to buffers

        buffer += token
        lower_buf += token.lower()

        out = []
        while True:
            if visible:

                # scan for START_TAG

                i = lower_buf.find(START_TAG)
                if i == -1:

                    # no START_TAG - emit

                    out.append(buffer)
                    buffer = ""
                    lower_buf = ""
                    break

                else:

                    # emit to tag, enter hidden mode

                    out.append(buffer[:i])
                    j = i + len(START_TAG)
                    buffer = buffer[j:]
                    lower_buf = lower_buf[j:]
                    visible = False
                    yield "__HIDDEN_ON__"
            else:

                # in hidden mode - scan for END_TAG

                i = lower_buf.find(END_TAG)
                if i == -1:
                    buffer = buffer[-64:]
                    lower_buf = lower_buf[-64:]
                    break
                else:

                    # drop hidden content until END_TAG + resume visible mode

                    j = i + len(END_TAG)
                    buffer = buffer[j:]
                    lower_buf = lower_buf[j:]
                    visible = True
                    yield "__HIDDEN_OFF__"

        # stream accumulated visible text

        chunk = "".join(out)
        if chunk:
            yield chunk

    # flush remaining visible text post-stream

    if visible and buffer:
        yield buffer

#    streamed_answer = ""
#    for token in llm.stream(prompt):
#        streamed_answer += token
#        yield token

#    yield "\n\n--- knowledge excerpts ---\n"
#    for i, doc in enumerate(docs):
#        yield f"\n[{i+1}] {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}"

# config streamlit ui

st.set_page_config(
    page_title = "mī lyte",
    page_icon = "🍂",
    layout = 'centered',
    )

st.write("**mī lyte** is a tool to provide knowledge on stress and resilience, recommending evidence-based informal mindfulness practices to build into your everyday life.\n\n**mī lyte** is _not_ a replacement for in-person therapy. It does not provide medical or psychiatric advice. ")

# aesthetics

with st.sidebar:
    st.title("🍂mī lyte")
    st.subheader('Accessible mindfulness skills for everyday life')
    st.write("\n\n")
    st.write("\n\n")
    img = Image.open('mheal_logo.png')
#    new_size = (300, 120)
#    img = img.resize(new_size)
    st.image(img, width = 120)
    st.caption("A prototype developed by [mHEAL: the Mindfulness for Health Equity Lab](https://sites.brown.edu/mheal/) at Brown University, grounded in _Mindfulness-Based Queer Resilience_ © Dr. Shufang Sun. ")

# initialize chat Hx

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# query input

user_input = st.chat_input("I could use some help with...")

# display previous exchanges

for user_q, assistant_r in st.session_state.chat_history:
    with st.chat_message('user'):
        st.markdown(user_q)
    with st.chat_message('assistant'):
        st.markdown(assistant_r)

# on new query

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_container = st.empty()
        visible_text = ""
        hidden = False ### are we inside <think>…</think>?


        for part in query_and_stream_ui(llm, retriever, user_input, prompt_template):

            # sentinel events

            if part == "__HIDDEN_ON__":
                hidden = True

                # render once with "_Generating..._"

                response_container.markdown(visible_text + " _Generating..._")
                continue
            if part == "__HIDDEN_OFF__":
                hidden = False

                # revert to cursor-on streaming immediately

                response_container.markdown(visible_text + "▍")
                continue

            # normal visible tokens

            visible_text += part

            # during visible streaming, show cursor "▍" 

            if not hidden:
                response_container.markdown(visible_text + "▍")
                time.sleep(0.015)
            else:

                # still hidden - display "_Generating..._"

                response_container.markdown(visible_text + " _Generating..._")

        # drop cursor on completion

        response_container.markdown(visible_text)

    # store (optional: save the final visible_text instead of "(see above)")
    
    st.session_state.chat_history.append((user_input, visible_text))