
from langchain_core.prompts import PromptTemplate

def query_and_stream(
    llm,
    retriever,
    query,
    prompt_template = None,
    show_sources = True,
    ):

    '''
    Wrapper that replicates RetrievalQA (chain_type = 'stuff') behavior with token streaming.

    Parameters:
    - llm: Ollama LLM (streamable, e.g. Deepseek-R1)
    - retriever: pre-specfied (external) vector store retriever
    - query: user question
    - prompt_template: optional PromptTemplate (context + question)
    - show_sources: determines whether to print the source documents after the answer
    '''

    # define PromptTemplate if none is passed

    if prompt_template is None:
        prompt_template = PromptTemplate(
            input_variables = ['context', 'question'],
            template = '''
    You are a knowledgable conversational agent that offers accurate, succinct, responses 
        based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:'''

        )

    # manually replicate RetrievalQA behavior (chain_type = 'stuff')

    #docs = retriever.get_relevant_documents(query) ### deprecated
    docs = retriever.invoke(query)

    if not docs:
        print("No relevant documents found.")
        return

    # concatenate ('stuff') context

    context = "\n\n".join(doc.page_content for doc in docs)

    # format prompt

    prompt = prompt_template.format(
        context = context, 
        question = query,
    )

    # stream response token-by-token

    print("\n🍂\n")
    for token in llm.stream(prompt):
        print(
            token, 
            end = "", 
            flush = True,
            )

    print("\n\n🍂")

    # print source metadata (optional)

    if show_sources:
        print("\nknowledge excerpts:\n")
        for i, doc in enumerate(docs):

        ### SJS 8/9: verbose w/ page_content...

            meta = doc.metadata
            print(f"--- excerpt {i+1} ---")
            print(f"metadata: {meta}")
            print(doc.page_content[:1000], "...\n")

        ### SJS 8/9: cleaner - metadata _only_

            #yield f"\n[{i+1}] {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}"       
