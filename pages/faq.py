import streamlit as st
import requests
import os

# Set up the API key and endpoint
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

def main():
    st.set_page_config(page_title="FAQs - WhatsUpDoc 🐇", layout="wide")
    # FAQ Section
    st.header("FAQs - WhatsUpDoc 🐇")

    # Expanders for each FAQ
    with st.expander("What is WhatsUpDoc?"):
        st.write("""
            ChatGPT is only trained on data up to a certain date. After that, it needs to look up information in an easy to consume format (for robots).
                 
            WhatsUpDoc takes out the hassle of keeping up with the documentation, Github Discussion, heck even Youtube videos so you always have a GPT with up-to-date information.
        """)

    with st.expander("How do I get started with WhatsUpDoc?"):
        st.write("""
            To get started, simply get a lifetime deal for $4.99 and validate your License Key. Bring your own OpenAI key and you're good to go!
        """)

    with st.expander("What kind of support is available?"):
        st.write("""
            Join the Discord community and I'll be there to help you out. You can also DM me on Twitter @0xyjkim
        """)

    with st.expander("Is there a trial period?"):
        st.write("""
            No, there is no trial period. It's only $4.99 for a lifetime deal. That means you get all future updates for free, forever. The people who decide to "wait and see" will FOMO in later at a higher price.
        """)

    with st.expander("What payment methods are accepted?"):
        st.write(f"""
            We accept a variety of payment methods including credit cards and PayPal.
            All transactions are secure and encrypted through LemonSqueezy.
        """)

    with st.expander("Can't I just create my own Custom GPT?"):
        st.write("""
            Yes, you can. However on ChatGPT you can only upload 10 files to your Custom GPT. On WhatsUpDoc, I went through the trouble of scraping, cleaning, loading, and indexing every single page of the documentation. I also went through the trouble of scraping Github Discussions and Forums and even Youtube videos.

            I'm perfectly aware that many of you are great developers and you could do this yourselves. But you already have enough on your plate. 
                 
            Why not let me continue to do the heavy lifting for you for just $4.99?
        """)
    # Deal link and product information
    deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/3ad0977c-3921-453b-aaa5-c94df765fe88'
    price_formatted = lemonsqueezy_product_data['attributes']['price_formatted']

    # Deal button
    st.markdown(f"<a href='{deal_link}' target='_blank'><button style='width: 100%; border-radius: 4px; background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Get Limited Lifetime Deal {price_formatted}</button></a>", unsafe_allow_html=True)
    st.caption("Limited Lifetime Deal is only available for a limited time. We will increase the price by $4.99 for every 5 integrations. Get it now before the price increases!")
if __name__ == "__main__":
    main()
