from sentence_transformers import SentenceTransformer
import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

#페이지 설정
st.set_page_config(
     page_title="ALL IS WELLNESS",
     page_icon=":dog:",
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': 'https://www.google.com',
         'Report a bug': "https://streamlit.io",
         'About': "YBIGTA"
     }
 )

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('total_embeded_without_reflection.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('All is Wellness 상담 챗봇')
st.markdown("Chatbot test")

if 'generated' not in st.session_state:
    st.session_state['generated'] = [] #챗봇 대화내용저장(generated 세션)

if 'past' not in st.session_state:
    st.session_state['past'] = [] #나의 대화저장(past세션) >> streamlit 재실행되어도 초기화x
    
#사용자 입력 form 생성

# 사용자의 input
with st.form('form', clear_on_submit=True): #submit 버튼 누를시 입력칸 초기화
    user_input = st.text_area('대화를 입력해주세요: ','',placeholder='시작하고 싶으시면 시작하기를, 끝내고 싶으면 그만하기를 입력해주세요')
    submitted = st.form_submit_button(label='보내기')

# 유저가 인풋 주고 전송버튼을 누른다면
if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

# 유저와 챗봇의 대화내역 저장
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['reflection_and_response'])

# 유저와 챗봇의 대화내역을 메시지형태로 표시해줌
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
        
# run할 때 주소 뒤에 --server.headless=true 붙이기