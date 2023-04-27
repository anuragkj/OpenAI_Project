import asyncio
import os
import pickle

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_chat import message

load_dotenv()
api_key = st.secrets["OPENAI_KEY"]  

# vectors = getDocEmbeds("gpt4.pdf")
# qa = ChatVectorDBChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), vectors, return_source_documents=True)

async def main():

    async def storeDocEmbeds(filename):
    
        
        corpus = st.session_state['stored_text']
        
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        vectors = await getDocEmbeds('random')
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4"), retriever=vectors.as_retriever(), return_source_documents=True)
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]

    if 'output' in st.session_state:
        llm = ChatOpenAI(model_name="gpt-4")
        chain = load_qa_chain(llm, chain_type="stuff")

        if 'history' not in st.session_state:
            st.session_state['history'] = []


        #Creating the chatbot interface
        with st.sidebar:   
        # Add the logo image to the sidebar
            image = Image.open("assets/images/feynmanai-no-bg.png")
            st.image(image)
            
            # Add the header to the sidebar
            st.header("Understanding complex topics made simple!")
            st.write("_Your very own personal tutor._")

        if 'ready' not in st.session_state:
            st.session_state['ready'] = False

        # uploaded_file = st.file_uploader("Choose a file (Generate report from Homepage if help required over the video)", type="pdf")

        # if st.button("Generate Chatbot"):
            
            # Add your code here that needs to be executed
            # pdf = PyPDF2.PdfFileReader()
            # qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

            # st.session_state['ready'] = True

        # st.divider()
        st.session_state['ready'] = True
        if st.session_state['ready']:

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Welcome! You can now ask any questions regarding the video"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey!"]

            # container for chat history
            response_container = st.container()

            # container for text box
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Query:", placeholder="e.g: Summarize the video in a few sentences", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    output = await conversational_chat(user_input)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
    else:
        with st.sidebar:   
        # Add the logo image to the sidebar
            image = Image.open("assets/images/feynmanai-no-bg.png")
            st.image(image)
            
            # Add the header to the sidebar
            st.header("Understanding complex topics made simple!")
            st.write("_Your very own personal tutor._")
        st.write("❗Please add a URL in home page first❗")


if __name__ == "__main__":
    asyncio.run(main())