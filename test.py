from pathlib import Path
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
import os
from langchain_community.vectorstores import Chroma
import chromadb
import chainlit as cl

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
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

#Initialize Azure OpenAI embeddings
    
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

#Initialize chroma
persist_directory = 'pdfDB'
client = chromadb.PersistentClient(path=persist_directory)
# db = Chroma(
#             collection_name="ssd_japanese_store",
#             embedding_function=embeddings,
#             client=client,
#             )



def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    print("Vector store start...")
    # Load PDFs and split into documents
    for pdf_path in pdf_directory.glob("*.pdf"):
        print(f"pdfs path ：{pdf_path}")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)
   
    # Convert text to embeddings
    db=Chroma.from_documents(docs, embeddings,collection_name="ssd_japanese_store",client=client,)
    print("Vector stored in Chroma index successfully.")
    # for doc in docs:
    #     print(f"doc ：{doc}")
    #     db.add_documents(documents=doc, embedding=embeddings)
    #     print("Vector stored in Pinecone index successfully.")
    print("There are", db._collection.count(), "in the collection")


process_pdfs(pdf_storage_path)

# store = {}
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

welcome_message = "Welcome to the Chainlit Pinecone demo! Ask anything about documents you vectorized and stored in your Pinecone DB."

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:\n{chat_history}
Follow Up Input: {question}
Standalone question:"""

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

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


# 学習問題の提供が要求される場合、
# 発言内容の形式としては、以下に回答の例のような学習問題を提供してください。
# #回答の例：
# 以下、用法が正しいのは選択してください。
# ア、部長、事務所におりますか？
# イ、部長、事務所にいらっしゃいますか？
# ウ、部長、事務所においですかウ?

# 提供した学習問題に対して、回答する際には、回答が正しいかの判断をしてください。また、関連知識点について説明してください。


from langchain.memory import  ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()
    persist_directory = 'pdfDB'
    client = chromadb.PersistentClient(path=persist_directory)

    vector_store = Chroma(
        collection_name="ssd_japanese_store",
        embedding_function=embeddings,
        client=client,
    )
    # contextualize_q_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", contextualize_q_system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ]
    # )
    # history_aware_retriever = create_history_aware_retriever(
    #     chat, vector_store.as_retriever(), contextualize_q_prompt
    # )

    # qa_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", qa_system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ]
    # )

    # question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
    # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
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
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm = chat,
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever(),
    #     memory=memory,
    #     return_source_documents=True,
    # )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8, "k":3}),
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

    cb = cl.AsyncLangchainCallbackHandler()

    # res = await chain.ainvoke({"input": message.content,},
    #                             config={
    #                                 "configurable": {"session_id": "abc123"}
    #                             },  # callbacks=[cb]
    #                         )
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
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

    await cl.Message(content=answer, elements=text_elements).send()