import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_path="data"):
    documents = []

    for root, _, files in os.walk(data_path):
        subject = root.split("/")[-1]

        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".pdf"):
                docs = PyPDFLoader(path).load()
            elif file.endswith(".docx"):
                docs = Docx2txtLoader(path).load()
            else:
                continue

            for d in docs:
                d.metadata["subject"] = subject
                d.metadata["source"] = file

            documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)
