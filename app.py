from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from replicate.client import Client
from flask import Flask, request, render_template,jsonify
app = Flask(__name__)
gptapi="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"
replicateapi="r8_0MD12IcK2Q2zsVGIoLnEk6vvUulxc6r01DynC"
chat_model = ChatOpenAI(model="gpt-4", temperature=0,openai_api_key=gptapi)
chat_model_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,openai_api_key=gptapi)


### download embeddings model
embeddings = embeddings = OpenAIEmbeddings(openai_api_key=gptapi)

### load vector DB embeddings
vectordb = FAISS.load_local(
   'vectordb/faiss_index_hp',
    embeddings,
    allow_dangerous_deserialization=True
)

prompt_template = """
Suppose you are a customer support agent of a UAE website.
Gave the answer in proper complete sentence and use only the following piece of context to gave answer and do not answer based on your own knowledge.

{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

retriever = vectordb.as_retriever(search_kwargs = {"k": 5, "search_type" : "similarity"})

gpt4Chain = RetrievalQA.from_chain_type(
    llm = chat_model,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever,
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)
gpt_3_5_chain = RetrievalQA.from_chain_type(
    llm = chat_model_3_5,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever,
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)
def GPT4(query):
    try:
      llm_response = gpt4Chain.invoke(query)
      ans = llm_response['result']
    except Exception as e:
        # Handle any errors and return the error message
        error_message = f"An error occurred: {str(e)}"
        return error_message,

    return ans
def GPT3_5(query):
    try:
      llm_response = gpt_3_5_chain.invoke(query)
      ans = llm_response['result']
    except Exception as e:
        # Handle any errors and return the error message
        error_message = f"An error occurred: {str(e)}"
        return error_message,

    return ans
prompt_template2 =  """
Suppose you are a customer support agent of a UAE website.
Gave the answer in proper complete sentence and use only the following piece of context to gave answer and do not answer based on your own knowledge.

{context}

The Question is asked below:
{question}
"""
replicate = Client(api_token=replicateapi)

PROMPT2 = PromptTemplate(
    template = prompt_template2,
    input_variables = ["context", "question"])

def Falcon(question):
    documents = vectordb.similarity_search(question)
    
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    prompt = PROMPT2.format(context=combined_context, question=question)
    falcon_response=replicate.run(
    "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
    input={
        "prompt": prompt,
        "temperature": 0.7,
        'max_new_tokens': 2048,
        },
    )
    suggestions = ''.join([str(s) for s in falcon_response])

    return suggestions

def Llama(question):
    documents = vectordb.similarity_search(question)
    
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    prompt = PROMPT2.format(context=combined_context, question=question)
    llama_response=replicate.run(
        "meta/llama-2-70b-chat",
        input={
            "prompt": prompt,
            "temperature": 0.75,
            'max_new_tokens': 2048,
        },
    )
    suggestions = ''.join([str(s) for s in llama_response])

    return suggestions

def calculate_rouge1_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    # Create a smoothing function
    smoothing_function = SmoothingFunction().method1
    
    # Calculate BLEU score
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
    
    return score
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    gpt_4_response = GPT4(user_query)
    gpt_3_5_response = GPT3_5(user_query)
    llama_response = Llama(user_query)
    falcon_response = Falcon(user_query)
    vectordb = FAISS.load_local(
    'vectordb/faiss_index_hp',
    embeddings,
    allow_dangerous_deserialization=True
    )
    documents = vectordb.similarity_search(user_query)
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    candidates=[gpt_4_response,gpt_3_5_response,falcon_response,llama_response]
    models=['GPT-3.5','GPT-4','Falcon','Llama-2'] 
    scores = [calculate_rouge1_score(combined_context, candidate) for candidate in candidates]
    print(scores)
    max_score_index = scores.index(max(scores))

    responses = {
        "GPT-4": gpt_4_response,
        "GPT-3.5": gpt_3_5_response,
        "LLaMA": llama_response,
        "Falcon": falcon_response,
        "Best Model":"The model with best performance is "+models[max_score_index]
    }
    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)