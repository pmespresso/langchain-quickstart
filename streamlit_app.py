import streamlit as st
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import LongContextReorder
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.retrievers import RePhraseQueryRetriever
from langchain.memory import ConversationBufferMemory
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_transformers import EmbeddingsRedundantFilter
from utils.stream_handler import StreamHandler


# Load environment variables
load_dotenv()

def setup_streamlit_page():
    st.set_page_config(page_title="🐇 WhatsUpDoc")
    st.title('🐇 WhatsUpDoc')
    st.sidebar.title('🐇 WhatsUpDoc')
    st.sidebar.markdown("""
    WhatsUpDoc is a **search engine** for your tech stack's documentation and broader knowledge base. 
    It uses **OpenAI** to understand your questions and **Langchain** to search your tech stack's documentation and broader knowledge base.
    """)


def initialize_globals():
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input('OpenAI API Key')
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings()
    return openai_api_key, supabase, embeddings


def configure_sidebar():
    with st.sidebar:
        st.markdown('This is a **demo** of the WhatsUpDoc app. Please enter your OpenAI API key below to get started.')
        st.markdown('The app is currently in **alpha** and is under active development.')
        st.divider()

        tech_stack_options = {
            'Tailwind CSS': 'tailwind_documents',
            'NextJS': 'nextjs_documents',
            'Stripe SDK': 'stripe_documents',
            'Langchain SDK': 'langchain_documents',
            'Streamlit SDK': 'streamlit_documents',
        }
        tech_stack = st.selectbox(
            label="Choose Tech", 
            help='Search the Documentation and Broader Knowledge Base of Your Tech Stack.',
            options=list(tech_stack_options.keys()), 
            label_visibility='visible'
        )

        st.divider()
        with st.form('waitlist_form'):
            st.text_input('Join the waitlist for early access to the app:', placeholder='e.g. harrisonchase@gmail.com')
            st.form_submit_button("Submit")

    return tech_stack_options[tech_stack]

def generate_response(input_text, chat_box):

  compression_retriever = get_retriever()
  qa_chain = get_qa_chain(compression_retriever, chat_box)

  result = qa_chain({ "question": input_text })

  st.markdown(result['answer'])


  if result['source_documents']:
    for document in result['source_documents']:
      st.markdown(document['page_content'])

def run_main_application(tech_stack, openai_api_key):
    st.title(tech_stack)
    with st.form('my_form'):
        text = st.text_area('Enter text:', 'How do I set a gradient from teal to blue to purple to a full page background?')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠️')
        if submitted and openai_api_key.startswith('sk-'):
            chat_box = st.empty()
            generate_response(text, chat_box)
 


def get_retriever():
    vectorstore = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        chunk_size=100,
        table_name=tech_stack,
        query_name=f"match_{tech_stack}",
    )

    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[
            relevant_filter,
            LongContextReorder()
        ]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, 
        base_retriever=vectorstore.as_retriever()
    )
    return compression_retriever


def get_qa_chain(compression_retriever: ContextualCompressionRetriever, chat_box):
  # Instantiate ConversationBufferMemory
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

  stream_handler = StreamHandler(chat_box, display_method='write')
  llm = ChatOpenAI(temperature=0.3, model='gpt-4-1106-preview', streaming=True, callbacks=[stream_handler])

  qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=compression_retriever, memory=memory, return_source_documents=True)

  return qa

# Setup Streamlit page configuration
setup_streamlit_page()

# Initialize global variables
openai_api_key, supabase_client, embeddings = initialize_globals()

# Sidebar configuration
tech_stack = configure_sidebar()

# Main application
run_main_application(tech_stack, openai_api_key)
