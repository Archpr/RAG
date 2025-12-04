import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


GOOGLE_API_KEY = "AIzaSyA1QukE3YvjSgL1PgANXb1CKM5Xynp1I_M"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = FAISS.load_local(
    "university_index_st",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # updated model
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_template(
    """Use the context to answer.
    If you don't know, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)




import streamlit as st

st.title("University Assistant")

q = st.text_input("Ask something")

if st.button("Submit"):
    resp = rag_chain.invoke(q)
    st.write(resp.content)

with open("queries.log", "a") as f:
    f.write(q + "\n")
