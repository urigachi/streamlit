pip install streamlit pyngrok
pip -q install langchain pypdf chromadb sentence-transformers faiss-gpu
pip install loguru
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
#from langchain.document_loaders import UnstructurePowerPointLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from google.colab import drive
from loguru import logger

import torch, locale
import warnings

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# 모델 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

model_id = "kyujinpy/Ko-PlatYi-6B"

# 전역 변수로 모델과 토크나이저 로드
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)
    st.session_state.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

def main(question, db):
    try:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        #inputs = tokenizer(question, return_tensors="pt")
        #outputs = model.generate(inputs.input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95)
        #answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text_generation_pipeline = pipeline(
             model = model,
             tokenizer = tokenizer,
             task = "text-generation",
             temperature = 0.2,
             return_full_text = True,
             max_new_tokens = 300,
        )

        prompt_template = """
        ### [INST]
        instruction:Answer the question based on your knowledge.
        Here is context to help:

        {context}
        ### question
        {question}
        [/INST]
        """

        koplatyi_llm = HuggingFacePipeline(pipeline = text_generation_pipeline)

        #Create prompt from prompt template
        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = prompt_template,
        )

        #Create llm chain
        llm_chain = LLMChain(llm = koplatyi_llm, prompt = prompt)
        #st.file_uploader를 사용하여 로컬에서 파일을 업로드하

        #with st.sidebar:
        #   upload_files = st.file_uploader("Upload your files", type = ['pdf', 'docx', 'ppt'],
        #   accept_multiple_files = True)
        #files_text = get_text(upload_files)

        #loader = PyPDFLoader("/content/drive/MyDrive/방송통신발전기본법.pdf")
        #pages = loader.load_and_split()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        #texts = text_splitter.split_documents(files_text)

        #model_name = "jhgan/ko-sbert-nli"
        #encode_kwargs = {'normalize_embeddings':True}
        #hf = HuggingFaceEmbeddings(
        #     model_name = model_name,
        #     encode_kwargs = encode_kwargs
        #)

        #db = FAISS.from_documents(texts, hf)
        retriever = db.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k':3}
        )

        rag_chain = (
         {"context":retriever, "question":RunnablePassthrough()}
           | llm_chain
        )

        warnings.filterwarnings('ignore')
        result = rag_chain.invoke(question)

        for i in result['context']:
            #st.write("주어진 근거":{i.page_content}/출처:{i.metadata['source']}-{i.metadata['page']})
            st.write(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']}-{i.metadata['page']}\n\n")

        st.write(f"\n답변:{result['text']}")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")

def get_text(docs):
    doc_list = [ ]
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
           loader = PyPDFLoader(file_name)
           documents = loader.load_and_split()
        elif '.docx' in doc.name:
           loader = Docx2textLoader(file_name)
           documents = loader.load_and_split()
       # elif '.pptx' in doc.name:
       #    loader = UnstructurePowerPointLoader(file_name)
       #    documnets = loader.load_and_split()
        doc_list.extend(documents)
        return doc_list

def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 100)
        chunks = text_splitter.split_documents(text)
        return chunks

def get_vectorstore(text_chunks):
    model_name = "jhgan/ko-sbert-nli"
    encode_kwargs = {'normalize_embeddings':True}
    hf = HuggingFaceEmbeddings(
             model_name = model_name,
             encode_kwargs = encode_kwargs
    )

    db = FAISS.from_documents(text_chunks, hf)
    return db

#def tiktoken_len(text):
#    tokenizer = tiktoken.get_encoding("cl100k_base")
#    tokens = tokenizer.encode(text)
#    return len(tokens)

if __name__ == "__main__":
    st.title('질문과 답변 시스템')
    with st.sidebar:
           upload_files = st.file_uploader("Upload your files", type = ['pdf', 'docx', 'ppt'],
           accept_multiple_files = True)
           process = st.button("process")
    files_text = get_text(upload_files)
    text_chunks = get_text_chunks(files_text)
    db = get_vectorstore(text_chunks)
    question = st.text_input('질문을 입력해주세요:')
    if st.button('답변 받기'):  # 사용자가 답변 받기 버튼을 클릭하면 main 함수 호출
        main(question, db)
