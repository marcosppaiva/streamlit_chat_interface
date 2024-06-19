import ollama
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

_MODELS = {
    'llama2': 'llama2',
    'llama3': 'llama3',
    'gemini-1.0': 'gemini-pro',
    'gemini-1.5': 'gemini-1.5-pro-latest',
}
_AVATARS = {'user': '🥸', 'assistant': '♠️'}


st.set_page_config(page_title='LLM Playground', initial_sidebar_state="expanded", layout='wide')


st.subheader('LLM Capgemini PlayGround', divider='blue', anchor=False)

selected_model = st.selectbox('Select the Model:', _MODELS, index=None)

if selected_model != None:
    model_type = _MODELS.get(selected_model, selected_model) # type: ignore

    if 'llama' in model_type:
        model = Ollama(model=model_type, base_url='http://54.216.99.70:11434')
    else:
        model = ChatVertexAI(model_name=model_type)


    message_container = st.container(height=500, border=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = _AVATARS.get(message['role'])
        with message_container.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])


    if prompt := st.chat_input('Enter a prompt here...'):
        try:
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            message_container.chat_message('user', avatar=_AVATARS.get('user')).markdown(prompt)

            with st.spinner('💡Thinking...'):
                response = model.invoke([HumanMessage(content=prompt)])
            
            st.session_state.messages.append({'role': 'assistant', 'content': response.content})

            message_container.chat_message('assistant', avatar=_AVATARS.get('assistant')).markdown(response.content)

            print(response)
        except Exception as err:
            print(err)

else:
    st.warning('Please, select a model')