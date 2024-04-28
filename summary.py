from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

# load_dotenv('var.env')

os.environ["OPENAI_API_KEY"] = "Your_api_key"
filePath = r"C:\Users\HP\OneDrive\Desktop\genai\crime-and-punishment.pdf"
loader = PyPDFLoader(filePath)
documents = loader.load()
print(f"length of docs : {len(documents)}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000, chunk_overlap=200)

docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)   

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)

template = """
INSTRUCTIONS:
Write a concise summary of the following content 

<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)


chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

while True:
    prompt = input("User>")
    if prompt.lower() == 'exit':
        break
    else:
        assistant_response = chain.invoke(prompt)
        print(f"AI Assistnat: {assistant_response['result']}")
        print("*********************************")
