import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai
import pandas as pd

load_dotenv()

def main():
    st.header("Course Planner")

    #adding temporary feature to test out different pdf files
    csv_file = st.file_uploader("Upload your CSV", type='csv')

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        text = "\n".join(df.values.astype(str).flatten())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = csv_file.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Enter your prompt")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            #model_name = "gpt-3.5-turbo"

            llm = ChatOpenAI(model_name="gpt-4-1106-preview")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
