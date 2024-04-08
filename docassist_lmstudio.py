from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chroma import Chroma
from langchain.text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Setup to use with LM Studio Local Server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load documents from URLs
loaders = [
    WebBaseLoader("https://durhamcollege.ca/academic-faculties/professional-and-part-time-learning/student-information/academic-policies-procedures"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# Set up Chroma for document embedding and search
embeddings = HuggingFaceEmbeddings()
persist_directory = './docs/chroma/'
vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()

def create_prompt_template(prompt):
    # Retrieve the most similar document's content based on the prompt
    context = vectordb.similarity_search(prompt, k=1)[0].page_content
    
    prompt_template = f"""[INST] <<SYS>>
Answer the following QUESTION based on the CONTEXT given. If you do not know the answer and the CONTEXT doesn't contain the answer truthfully say "I don't know".<</SYS>>
CONTEXT: {context}

QUESTION: {prompt}[/INST]
"""
    return prompt_template

# Example usage
prompt = "What is the limit of the attachable files?"
prompt_template = create_prompt_template(prompt)

# Send the constructed prompt template to LM Studio for completion
completion = client.chat.completions.create(
    model="TheBloke/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q6_K.gguf",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": "Generate response based on context."}
    ],
    temperature=0.7,
)

# Output the response
print(completion.choices[0].message)
