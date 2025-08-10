
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_retrieval_qa_chain(
    llm,
    retriever,
    chain_type = 'stuff',
    *,
    prompt = None,
    question_prompt = None,
    refine_prompt = None,
    combine_prompt = None,
    ):
    
    '''
    Build a RetrievalQA chain with support for external prompt injection.

    Parameters:
    - llm: the LLM object (e.g., Ollama instance)
    - retriever: a LangChain retriever (e.g., FAISS)
    - chain_type: 'stuff', 'map_reduce', or 'refine'
    - prompt: for 'stuff' chains
    - question_prompt & refine_prompt: for 'refine' chains
    - question_prompt & combine_prompt: for 'map_reduce' chains
    '''

    if chain_type == 'stuff':
        prompt = prompt or PromptTemplate(
            input_variables=['context', 'question'],
            template = '''
    You are a knowledgable conversational agent that offers accurate, succinct, responses 
        based on the provided context.

    Context:
    {context}

    Question:
    {question}
    '''
        )
        
        chain_type_kwargs = {'prompt': prompt}

    elif chain_type == 'map_reduce':
        question_prompt = question_prompt or PromptTemplate(
            input_variables = ['context', 'question'],
            template = '''
    Examine the following context to respond to the query as accurately as possible.

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''
        )
        
        combine_prompt = combine_prompt or PromptTemplate(
            input_variables = ['summaries', 'question'],
            template = '''
    You are a knowledgable conversational agent that synthesizes multiple responses to 
        create a single comprehensive response.

    Summaries:
    {summaries}

    Question:
    {question}

    Final Answer:
    '''
        )
        
        chain_type_kwargs = {
            'question_prompt': question_prompt,
            'combine_prompt': combine_prompt,
            }

    elif chain_type == 'refine':
        question_prompt = question_prompt or PromptTemplate(
            input_variables = ['context', 'question'],
            template = '''
    You are a knowledgable conversational agent that offers accurate, succinct, responses 
        based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''
        )
        
        refine_prompt = refine_prompt or PromptTemplate(
            input_variables = ['context', 'question', 'existing_answer'],
            template = '''
        You are improving an existing response using new context.

        Existing Answer:
        {existing_answer}

        New Context:
        {context}

        Question:
        {question}

        Refined Answer:
        '''
        )
        chain_type_kwargs = {
            'question_prompt': question_prompt,
            'refine_prompt': refine_prompt
        }

    else:
        raise ValueError(f"Unsupported chain_type: {chain_type}")

    return RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = chain_type,
        return_source_documents = True,
        chain_type_kwargs = chain_type_kwargs
        )

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def query_and_stream(
    llm,
    retriever,
    query,
    prompt_template = None,
    show_sources = True,
    ):
    
    '''
    Wrapper that replicates RetrievalQA (chain_type='stuff') behavior with token streaming.

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
    
    print("\n🔮\n")
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
