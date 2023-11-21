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

# Function to call the API for license key verification
def verify_license_key(license_key):
    url = "YOUR_API_ENDPOINT"  # Replace with your API endpoint
    data = {"license_key": license_key}
    response = requests.post(url, json=data)
    return response.ok  # or any other logic based on your API response


def setup_streamlit_page():
    st.set_page_config(page_title="🐇 WhatsUpDoc")
    st.sidebar.title('🐇 WhatsUpDoc')
    st.sidebar.markdown("""
    WhatsUpDoc is a **search engine** for your tech stack's documentation and broader knowledge base. 
    It uses **OpenAI** to understand your questions and **Langchain** to search your tech stack's documentation and broader knowledge base.
    """)


def initialize_globals():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings()
    return openai_api_key, supabase, embeddings


def configure_sidebar():
    with st.sidebar:
        st.markdown('The app is currently in **alpha**.')
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
        
        # Form for license key input
        with st.form("license_key_form"):
            license_key = st.text_input("License Key", "")
            submitted = st.form_submit_button("Verify")

            if submitted:
                if verify_license_key(license_key):
                    st.success("Verified License Key")
                    verified = True
                else:
                    st.error("Invalid License Key")
                    verified = False
            else:
                verified = False
        
        # Display the button for the deal only if the license key is verified
        if not verified:
            deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/7a80a616-ef60-4fe6-9ff0-d67acacc8ab0'
            st.markdown(f"<a href='{deal_link}' target='_blank'><button style='background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Get Limited Lifetime Deal $29.99</button></a>", unsafe_allow_html=True)


        if not openai_api_key:
          st.divider()
          st.markdown('This is a **demo** of the WhatsUpDoc app. Please enter your OpenAI API key below to get started.')
          st.sidebar.text_input('OpenAI API Key')

    return tech_stack_options[tech_stack]

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
            # Access attributes directly
            # page_content = document.page_content
            metadata = document.metadata

            # If you need to access specific metadata attributes
            if metadata:
                # location = metadata.get('loc', 'No location provided')
                source = metadata.get('source', 'No source provided')

                # Use location and source as needed
                # st.markdown(f"Location: {location}")
                st.markdown(f"{source}")

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
