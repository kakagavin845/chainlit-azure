from pathlib import Path
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os
import asyncio
from langchain_community.vectorstores import Chroma
import chromadb
import chainlit as cl

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)

from langchain.globals import set_debug
set_debug(True)

chunk_size = 1024
chunk_overlap = 50
pdf_storage_path = "./pdfs"

# Load environment variables
load_dotenv()

# OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ADA_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_ADA_DEPLOYMENT_VERSION
)

chat = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
    openai_api_type="azure",
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    streaming=True
)

# Initialize chroma
persist_directory = 'pdfDB'
client = chromadb.PersistentClient(path=persist_directory)

def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    print("Vector store start...")
    # Delete existing collection if it exists
    try:
        print(client.list_collections())
        print("Deleting existing vector store...")
        client.delete_collection(name="ssd_japanese_store")
        print("Existing vector store deleted.")
    except ValueError:
        print("No existing vector store found.")

    # Load PDFs in batches and process them
    pdf_paths = list(pdf_directory.glob("*.pdf"))
    batch_size = 4
    for i in range(0, len(pdf_paths), batch_size):
        batch_paths = pdf_paths[i:i + batch_size]
        docs = []
        for pdf_path in batch_paths:
            print(f"Processing PDF: {pdf_path}")
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()
            docs += text_splitter.split_documents(documents)
        
        # Convert text to embeddings and store in Chroma
        Chroma.from_documents(docs, embeddings, collection_name="ssd_japanese_store", client=client)
        print(f"Batch {i // batch_size + 1} vectorized and stored in Chroma.")

    print("All PDFs processed and stored in Chroma index.")
    vector_store = client.get_collection(name="ssd_japanese_store")
    print("There are", vector_store.count(), "documents in the collection")

# process_pdfs(pdf_storage_path)

welcome_message = "初めまして！私は日本語学習のお手伝いをする、あなたの日本語学習アシスタントです。日本語の質問や疑問があれば、どんなことでも気軽に聞いてくださいね。一緒に楽しく学んでいきましょう！"

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:\n{chat_history}
Follow Up Input: {question}
Standalone question:"""

qa_system_prompt = """
あなたは、日本語の学習問題を提供する優秀なチャットボットです。
日本語学習に関する質問にのみ答え、日本語学習に関連しない質問には丁寧にお断りします。

以下にお客様とAIの親し気な会話履歴を提示しますので、それに基づいて発言しなさい。
発言内容はコンテキスト情報を参照して回答してください。コンテキスト情報にない場合は事前学習されたデータも使って回答してください。
'''
#会話履歴:

{chat_history}
'''

'''
#コンテキスト情報:
      
{context}
'''
"""

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

@cl.on_chat_start
async def start():

    msg = cl.Message(content="",disable_feedback=True)
    for char in welcome_message:
        # Send each character with a delay
        await msg.stream_token(char)
        if char != ' ':  # Add a longer delay after non-space characters
            await asyncio.sleep(0.05)  # Adjust the delay time as needed
        else:
            await asyncio.sleep(0.01)  # Shorter delay after spaces

    await msg.send()

    # await cl.Message(content=welcome_message).send()  # Send welcome message
    persist_directory = 'pdfDB'
    client = chromadb.PersistentClient(path=persist_directory)

    vector_store = Chroma(
        collection_name="ssd_japanese_store",
        embedding_function=embeddings,
        client=client,
    )
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    condense_question_prompt = PromptTemplate(
        template=condense_question_template,
        input_variables=["chat_history", "question"]
    )

    human_prompt = "{question}"

    messages = [
        SystemMessagePromptTemplate.from_template(qa_system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8, "k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_question_prompt,
        condense_question_llm=chat,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    res = await chain.ainvoke(message.content)
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=text_elements).send()
    msg = cl.Message(content="", elements=text_elements)
    for char in answer:
        # Send each character with a delay
        await msg.stream_token(char)
        if char != ' ':  # Add a longer delay after non-space characters
            await asyncio.sleep(0.05)  # Adjust the delay time as needed
        else:
            await asyncio.sleep(0.01)  # Shorter delay after spaces

    await msg.send()
