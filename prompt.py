from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-Ir2mEo4OFYteUFwsdfXeT3BlbkFJBwoOV6Hgv1WLGjTSy2kA"

embeddings = OpenAIEmbeddings()

loader = PyPDFLoader(r"D:\Cogent-lab-task\crime-and-punishment.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


template = """
INSTRUCTIONS:

Act as an educational planner with access to comprehensive data about individual students. Analyze the diverse information available, which includes their academic performance in various subjects, learning preferences (visual, auditory, kinesthetic), involvement in extracurricular activities, and specific goals or challenges (like preparing for an upcoming exam or addressing learning disabilities).

REMEMBER:

1 - Tailored Study Plan Development:
Create personalized study plans for each student based on individual learning styles and performance analysis. These plans should optimize study hours and techniques, with targeted strategies for improving weaker subjects and enhancing skills in areas of strength.
2 - Integration of Goals and Activities:
Ensure that the study plan aligns with each student's unique needs, aspirations, extracurricular commitments, and personal goals. It should provide a balanced approach that promotes academic success while supporting overall well-being and personal development.
3- Continuous Evaluation and Adjustment:
Establish a method for ongoing assessment of the student's progress with the study plan. Regular reviews will allow for adjustments to be made to the plan to better meet the evolving educational needs and aspirations of the student.

Note Make sure format it accordingly using lists, sublists, headings, sub headings and bullet points.
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

model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
prompt_template = PromptTemplate(input_variables=["history", "context", "question"],
                                 template=template)
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever,  # Use the instance variable here
    chain_type_kwargs={"verbose": False, "prompt": prompt_template,
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
)


print("Chat Assistant \n")

while True:
    print("\n******************************************************************")
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! Bye Bye.")
        break
    else:
        response_dict = chain.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
    print("******************************************************************\n")
