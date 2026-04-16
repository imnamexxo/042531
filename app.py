import os
import streamlit as st
from dotenv import load_dotenv
os.environ["OPENAI_API_KEY"] = "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"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


# 환경 변수 로드
api_key = os.getenv("OPENAI_API_KEY")

# 세션별 채팅 기록 저장소
store = {}

# FAISS 저장 경로
FAISS_DIRECTORY = "./faiss_index"

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 최근 4개 발화만 사용
def get_trimmed_history(session_id: str, max_messages: int = 4):
    history = get_session_history(session_id)
    return history.messages[-max_messages:]


# PDF 처리 함수
@st.cache_resource
def process_pdf():
    pdf_path = "../data/2024_KB_부동산_보고서_최종.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split = text_splitter.split_documents(documents)
    return split


# FAISS vectorstore 초기화
@st.cache_resource
def initialize_vectorstore():
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    if os.path.exists(FAISS_DIRECTORY):
       faiss_db = FAISS.load_local(
           FAISS_DIRECTORY,
           embedding_function,
           allow_dangerous_deserialization=True
        )
    else:
        split = process_pdf()
        faiss_db = FAISS.from_documents(documents=split, embedding=embedding_function)
        faiss_db.save_local(FAISS_DIRECTORY)

    return faiss_db


# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다.
다음 컨텍스트를 바탕으로 사용자의 질문에 한국어로 정확하고 간결하게 답변해주세요.
모르는 내용은 추측하지 말고, 보고서에서 확인되지 않는다고 말해주세요.

컨텍스트:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_chain(question: str, session_id: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # 최근 4개 발화만 chat_history로 전달
        trimmed_history = get_trimmed_history(session_id, max_messages=4)

        chain = prompt | model | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": trimmed_history
        })

        return response

    return run_chain


# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    session_id = "streamlit_session"

    # 기존 채팅 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("부동산 관련 질문을 입력하세요")

    if user_input:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain(user_input, session_id)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 전체 대화 기록 저장
        history = get_session_history(session_id)
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=response))


if __name__ == "__main__":
    main()