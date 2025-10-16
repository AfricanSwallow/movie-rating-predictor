"""
Example: How to integrate review_clustering_page.py into a multi-page app

This is a complete working example your friend can reference.
"""

import streamlit as st

# Page configuration (do this ONCE at the top)
st.set_page_config(
    page_title="Movie Analysis Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🎬 Movie Analysis Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "**Navigation**",
    [
        "🏠 Home",
        "🎬 Review Clustering Analysis",
        "📊 Box Office Predictions", 
        "🎭 Genre Trends",
        "⭐ Rating Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Select a page above to explore different analyses")

# Page routing
if page == "🏠 Home":
    st.title("🎬 Welcome to Movie Analysis Platform")
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Available Analyses
    
    Use the sidebar to navigate between different analysis pages:
    
    - **🎬 Review Clustering Analysis** - Understanding what viewers value in movies
    - **📊 Box Office Predictions** - Revenue and profit forecasting
    - **🎭 Genre Trends** - Genre popularity over time
    - **⭐ Rating Analysis** - Rating patterns and predictions
    """)
    
    # Show some summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies Analyzed", "10,000+")
    with col2:
        st.metric("Reviews Processed", "75,000+")
    with col3:
        st.metric("Genres Covered", "20+")

elif page == "🎬 Review Clustering Analysis":
    # THIS IS WHERE YOU INTEGRATE THE REVIEW CLUSTERING PAGE
    from review_clustering_page import render_review_clustering_page
    render_review_clustering_page()
    
    # That's it! The page handles everything else.

elif page == "📊 Box Office Predictions":
    st.title("📊 Box Office Predictions")
    st.info("This is your friend's existing box office prediction page")
    # Their existing code here...
    
elif page == "🎭 Genre Trends":
    st.title("🎭 Genre Trends Analysis")
    st.info("This is your friend's existing genre trends page")
    # Their existing code here...
    
elif page == "⭐ Rating Analysis":
    st.title("⭐ Rating Analysis")
    st.info("This is your friend's existing rating analysis page")
    # Their existing code here...

# Footer (optional)
st.sidebar.markdown("---")
st.sidebar.markdown("**Movie Analysis Platform** v1.0")
st.sidebar.markdown("Built with Streamlit 🎈")
