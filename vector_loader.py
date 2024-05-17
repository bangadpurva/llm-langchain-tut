#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        API_KEY = "" #copy OPENAI key generated
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        loader=PyPDFLoader("./some.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=API_KEY))
        vectorstore.save_local(DB_FAISS_PATH)