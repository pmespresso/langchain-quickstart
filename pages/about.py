import streamlit as st

def main():
    st.set_page_config(page_title="About WhatsUpDoc 🐇", layout="wide")
    st.title("About WhatsUpDoc 🐇")

    st.markdown("""
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

    deal_link = 'https://whatsupdoc.lemonsqueezy.com/checkout/buy/7a80a616-ef60-4fe6-9ff0-d67acacc8ab0'
    st.markdown(f"<a href='{deal_link}' target='_blank'><button style='background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;'>👉 Get Limited Lifetime Deal $9.99</button></a>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
