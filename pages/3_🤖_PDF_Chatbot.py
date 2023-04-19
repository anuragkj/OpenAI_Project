from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
from PIL import Image

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  

# vectors = getDocEmbeds("gpt4.pdf")
# qa = ChatVectorDBChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), vectors, return_source_documents=True)

async def main():

    async def storeDocEmbeds(file, filename):
    
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
        
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(file, filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history_pdf']})
        st.session_state['history_pdf'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]


    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history_pdf' not in st.session_state:
        st.session_state['history_pdf'] = []


    #Creating the chatbot interface
    with st.sidebar:   
    # Add the logo image to the sidebar
        image = Image.open("assets/images/feynmanai-no-bg.png")
        st.image(image)
        
        # Add the header to the sidebar
        st.header("Understanding complex topics made simple!")
        st.write("_For students with special needs._")

    if 'ready_pdf' not in st.session_state:
        st.session_state['ready_pdf'] = False

    uploaded_file = st.file_uploader("Choose a file (Generate report from Homepage if help required over the video)", type="pdf")

    if uploaded_file is not None:

        with st.spinner("Processing..."):
        # Add your code here that needs to be executed
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # pdf = PyPDF2.PdfFileReader()
            vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)
            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

        st.session_state['ready_pdf'] = True

    # st.divider()

    if st.session_state['ready_pdf']:

        if 'generated_pdf' not in st.session_state:
            st.session_state['generated_pdf'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

        if 'past_pdf' not in st.session_state:
            st.session_state['past_pdf'] = ["Hey!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past_pdf'].append(user_input)
                st.session_state['generated_pdf'].append(output)

        if st.session_state['generated_pdf']:
            with response_container:
                for i in range(len(st.session_state['generated_pdf'])):
                    message(st.session_state["past_pdf"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated_pdf"][i], key=str(i), avatar_style="fun-emoji")


if __name__ == "__main__":
    asyncio.run(main())