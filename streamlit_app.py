import streamlit as st
import requests
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

# Initialize the session state for verification status
if 'verified' not in st.session_state:
    st.session_state.verified = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# Function to call the API for license key verification
def verify_license_key(license_key):
    url = "https://api.lemonsqueezy.com/v1/licenses/validate"
    headers = {"Accept": "application/json"}
    data = {"license_key": license_key}
    response = requests.post(url, headers=headers, data=data)
    # Assuming a successful response indicates a valid license key
    if response.ok:
        st.session_state.verified = True
    else:
        st.session_state.verified = False
    return response.ok

def setup_streamlit_page():
    st.set_page_config(page_title="WhatsUpDoc 🐇")
    st.sidebar.title('WhatsUpDoc 🐇')
    st.sidebar.markdown("""
    WhatsUpDoc is a **search engine** for your tech stack's documentation and broader knowledge base. 
    It uses **OpenAI** to understand your questions and **Langchain** to search your tech stack's documentation and broader knowledge base.
    """)


def initialize_globals():
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.openai_api_key
    verified = st.session_state.verified
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings()
    return openai_api_key, supabase, embeddings, verified


def configure_sidebar():
    with st.sidebar:
        st.markdown('The app is currently in **alpha**.')
        st.divider()

        tech_stack_options = {
            'Tailwind CSS': {
                'table_name': 'tailwind_documents',
                'heading': 'Tailwind CSS Documentation + Github Discussions',
                'placeholder_query': 'How do I set a gradient from teal to blue to purple to a full page background in Tailwind?'
            },
            'NextJS': {
                'table_name': 'nextjs_documents',
                'heading': 'NextJS Documentation + Github Discussions',
                'placeholder_query': 'Can you help me migrate my app from page router to apps router in NextJS?'
            },
            'Stripe SDK': {
                'table_name': 'stripe_documents',
                'heading': 'Stripe SDK Documentation + Stripe Guides',
                'placeholder_query': 'How do I create a customer with Stripe?'
            },
            'Langchain SDK': {
                'table_name': 'langchain_documents',
                'heading': 'Langchain SDK Documentation + Github Discussions',
                'placeholder_query': 'How do I do RAG with Chroma using Langchain?'
            },
            'Streamlit SDK': {
                'table_name': 'streamlit_documents',
                'heading': 'Streamlit SDK Documentation + Discuss Forum',
                'placeholder_query': 'How do I create an LLM app with Streamlit?'
            },
        }
        tech_stack = st.selectbox(
            label="Choose Tech", 
            help='Search the Documentation and Broader Knowledge Base of Your Tech Stack.',
            options=list(tech_stack_options.keys()), 
            label_visibility='visible'
        )

        st.divider()

        # Display the verification message if the license key is already verified
        if st.session_state.get('verified', False):
            st.success("License Key Verified")
        else:
          # Form for license key input
          with st.form("license_key_form"):
              license_key = st.text_input("License Key", "")
              submitted = st.form_submit_button("Verify")

              if submitted:
                  if verify_license_key(license_key):
                      st.session_state.verified = True
                      st.success("License Key Verified")
                  else:
                      st.session_state.verified = False
                      st.error("Invalid License Key")
                      
        # Display the button for the deal only if the license key is verified
        if not st.session_state.verified:
            deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/7a80a616-ef60-4fe6-9ff0-d67acacc8ab0'
            st.markdown(f"<a href='{deal_link}' target='_blank'><button style='background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Get Limited Lifetime Deal $29.99</button></a>", unsafe_allow_html=True)

        # If OpenAI API Key is not yet set, display the form to set it
        if not st.session_state.openai_api_key:
            with st.form("openai_api_key_form"):
              st.markdown('Please enter your OpenAI API key below to get started.')
              openai_api_key = st.text_input("OpenAI API Key", "")
              submit_openai_key_button = st.form_submit_button("Submit")

              if submit_openai_key_button:
                  st.session_state.openai_api_key = openai_api_key
                  st.success("OpenAI API Key is Set")
              else:
                  st.session_state.openai_api_key = None
                  st.error("OpenAI API Key is Not Set")
        else:
          st.success("OpenAI API Key is Set")


    return tech_stack_options[tech_stack]

def get_retriever():
    vectorstore = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        chunk_size=100,
        table_name=tech_stack['table_name'],
        query_name=f"match_{tech_stack['table_name']}",
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

  stream_handler = StreamHandler(chat_box, display_method='markdown')
  llm = ChatOpenAI(temperature=0.3, model='gpt-4-1106-preview', streaming=True, callbacks=[stream_handler])

  qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=compression_retriever, memory=memory, return_source_documents=True)

  return qa

def generate_response(input_text, chat_box):
  compression_retriever = get_retriever()
  qa_chain = get_qa_chain(compression_retriever, chat_box)

  result = qa_chain({ "question": input_text })

  st.markdown(result['answer'])

  print (result.keys())

  source_documents = result.get('source_documents')

  if source_documents:
    st.header("Sources")
    for document in source_documents:
        if document:
            # page_content = document.page_content
            metadata = document.metadata

            # If you need to access specific metadata attributes
            if metadata:
                source = metadata.get('source', 'No source provided')
                st.markdown(f"{source}")

def run_main_application(tech_stack, verified, openai_api_key):
    st.title(tech_stack['heading'])

    # Check if the license key is verified
    if not verified:
        st.warning("Please verify your license key before using the app!", icon="⚠️")
        st.markdown("Go to the sidebar to enter and verify your license key.")
    else:
      with st.form('main'):
          text = st.text_area('Enter text:', tech_stack['placeholder_query'])
          submitted = st.form_submit_button('Submit')
          if not openai_api_key or not openai_api_key.startswith('sk-'):
              st.warning('Please enter your OpenAI API key!', icon='⚠️')
          if submitted and openai_api_key and openai_api_key.startswith('sk-'):
              chat_box = st.empty()
              generate_response(text, chat_box)


# Setup Streamlit page configuration
setup_streamlit_page()

# Initialize global variables
openai_api_key, supabase_client, embeddings, verified = initialize_globals()

# Sidebar configuration
tech_stack = configure_sidebar()

# Main application
run_main_application(tech_stack, verified, openai_api_key)
