import streamlit as st
import requests
import os
from dotenv import load_dotenv
# from langchain.schema import HumanMessage
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import LongContextReorder
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.retrievers import RePhraseQueryRetriever
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

lemonsqueezy_api_key = os.getenv("LEMONSQUEEZY_API_KEY")
lemonsqueezy_product_endpoint = "https://api.lemonsqueezy.com/v1/products/150868"
headers = {
    'Accept': 'application/vnd.api+json',
    'Content-Type': 'application/vnd.api+json',
    'Authorization': f'Bearer {lemonsqueezy_api_key}'
}

# Get the product information from the LemonSqueezy API
response = requests.get(lemonsqueezy_product_endpoint, headers=headers)

lemonsqueezy_product_data = response.json().get('data')

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
    # Set page title and favicon
    st.set_page_config(page_title="WhatsUpDoc 🐇", page_icon="🐇")

    # Define your meta tags and OG tags
    meta_tags_html = """
    <head>
    <meta charset="UTF-8">
    <title>🐇 WhatsUpDoc - Talk to Your Docs, Discussions, Knowledgebase. More Up to Date than ChatGPT. Links to Sources. </title>
    <link rel="icon" href="https://github.com/pmespresso/gitdraft-dashboard/assets/10432070/c56678c6-d0bb-4097-a792-a1d2b6d92f12" />
    <link
        rel="icon"
        type="image/png"
        sizes="32x32"
        href="/favicon-32x32.png"
    />
    <link
        rel="icon"
        type="image/png"
        sizes="16x16"
        href="/favicon-16x16.png"
    />
    <link
        rel="apple-touch-icon"
        sizes="180x180"
        href="/apple-touch-icon.png"
    />
    <meta
        name="description"
        content="🐇 WhatsUpDoc - Talk to Your Docs, Discussions, Knowledgebase. More Up to Date than ChatGPT. Links to Sources."
    />
    <meta
        name="keywords"
        content="WhatsUpDoc,Documentation,Search Engine,Tech Stack,Knowledge Base, Github, OpenAI, AI, Talk to Docs"
    />
    <meta name="description" content="🐇 WhatsUpDoc - Talk to Your Tech Documentation, Github Discussion, and Extended Knowledge Base. More Up to Date than ChatGPT.">
    <meta name="keywords" content="WhatsUpDoc,Documentation,Search Engine,Tech Stack,Knowledge Base, Github, OpenAI, AI">
    <meta name="author" content="0xyjkim">
    <meta property="og:title" content="🐇 WhatsUpDoc" />
    <meta property="og:type" content="website" />
    <meta property="og:description" content="🐇 WhatsUpDoc - Talk to Your Tech Documentation, Github Discussion, and Extended Knowledge Base. More Up to Date than ChatGPT." />
    <meta property="og:image" content="https://github.com/pmespresso/whatsupdoc/assets/10432070/9a8bceee-869b-498a-b419-75b9930cb403" />
    <meta property="og:url" content="https://www.whatsupdoc.dev" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="https://whatsupdoc.dev" />

    <meta name="twitter:site" content="@0xyjkim" />
    <meta name="twitter:creator" content="@0xyjkim" />
    <meta
        property="twitter:image"
        content="https://github.com/pmespresso/whatsupdoc/assets/10432070/9a8bceee-869b-498a-b419-75b9930cb403"
    />
    <meta
        property="twitter:title"
        content="🐇 WhatsUpDoc"
    />
    <meta
        property="twitter:description"
        content="Talk to Your Tech Documentation, Github Discussion, and Extended Knowledge Base. More Up to Date than ChatGPT."
    />
    </head>
    """

    # Use st.markdown to insert HTML into the Streamlit app
    st.markdown(meta_tags_html, unsafe_allow_html=True)
            # Display the button for the deal only if the license key is verified
    if not st.session_state.verified:
        deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/3ad0977c-3921-453b-aaa5-c94df765fe88'
        st.markdown(f"<a href='{deal_link}' target='_blank'><button style='width: 100%; border-radius: 4px; background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Pay Once, Use Forever {lemonsqueezy_product_data['attributes']['price_formatted']}</button></a>", unsafe_allow_html=True)
        st.caption("Limited Lifetime Deal is only available for a limited time. :red[We will increase the price by $4.99 for every 5 integrations]. Get it now before the price increases!")

    st.sidebar.title('WhatsUpDoc 🐇')
    st.sidebar.markdown("""
    Chat with your tech stack's up-to-date documentation and broader knowledge base (including Github Discussions, Forums, etc.).
    """)


def initialize_globals():
    openai_api_key = st.session_state.openai_api_key
    verified = st.session_state.verified
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return openai_api_key, supabase, embeddings, verified


def configure_sidebar():
    with st.sidebar:
        st.markdown('The app is currently in **alpha**.')
        st.markdown('[Learn more about WhatsUpDoc](/about).')
        st.divider()

        tech_stack_options = {
            'Langchain Python SDK': {
                'table_name': 'langchain_documents',
                'heading': 'Langchain Python SDK Documentation + Github Discussions',
                'placeholder_query': 'How do I do RAG with Chroma using Langchain?'
            },
            'Streamlit SDK': {
                'table_name': 'streamlit_documents',
                'heading': 'Streamlit SDK Documentation + Discuss Forum',
                'placeholder_query': 'How do I create an LLM app with Streamlit?'
            },
            'PaddleJS': {
                'table_name': 'paddle_documents',
                'heading': 'PaddleJS Documentation',
                'placeholder_query': 'How do I create a checkout overlay with PaddleJS?'
            },
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
        }
        tech_stack = st.selectbox(
            label="Choose Tech", 
            help='Search the Documentation and Broader Knowledge Base of Your Tech Stack.',
            options=list(tech_stack_options.keys()), 
            label_visibility='visible'
        )

        st.divider()

        # Display the verification message if license key is already verified
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
        st.caption("Vote For the Next Integration 👇")
        st.link_button("Join our Discord 👾", url='https://discord.gg/VjEhmn2h')
        


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
        st.warning("Please verify your license key before using the app.", icon="⚠️")
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
