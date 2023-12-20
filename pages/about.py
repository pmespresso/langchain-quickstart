import streamlit as st
import requests
import os

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
    st.set_page_config(page_title="About WhatsUpDoc 🐇", layout="wide")
    st.title("About WhatsUpDoc 🐇")

    st.markdown("""
        Chat with your tech stack's up-to-date documentation and broader knowledge base (including Github
            Discussions, Forums, etc.).
        ## The Problem
        Navigating tech documentation is like finding a needle in a haystack. Developers waste hours in this maze. Existing AI tools? Outdated or unspecific. The result? Frustration.

        ## WhatsUpDoc at a Glance
        WhatsUpDoc changes the game. It's a chat-based portal to a universe of tech docs - think Next.js, Tailwind, Stripe SDK. 
                
        **Our twist? We include forums and GitHub Discussions.**
                
        Your benefit? Accurate, comprehensive answers, fast. Pay once, use forever. Just bring your OpenAI key.

        Enter WhatsUpDoc: a chat interface to a world of tech docs, from Next.js to Stripe SDK. We go beyond, integrating forums and GitHub Discussions for a fuller picture. Pay once, use forever. Just add your OpenAI key.

        ## The Solution
        WhatsUpDoc turns the chaos of information into clarity. Precise, comprehensive answers in a snap. It's like your always-on, know-it-all dev companion.

        ## Growing Knowledge Base
        Our knowledge base is not just vast; it's constantly evolving. Depth, breadth, latest updates - we're expanding every day. **Act now, and get grandfathered into a world of ever-growing tech insights.**
    """)

    deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/3ad0977c-3921-453b-aaa5-c94df765fe88'

    st.markdown(f"<a href='{deal_link}' target='_blank'><button style='width: 100%; border-radius: 4px; background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Get Limited Lifetime Deal {lemonsqueezy_product_data['attributes']['price_formatted']}</button></a>", unsafe_allow_html=True)
    st.caption("Limited Lifetime Deal is only available for a limited time. :red[We will increase the price by $4.99 for every 5 integrations]. Get it now before the price increases!")

if __name__ == "__main__":
    main()
