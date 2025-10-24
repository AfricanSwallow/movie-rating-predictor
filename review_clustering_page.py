"""
Review Clustering Analysis Page for Streamlit Multi-Page App

This is a single-page component that can be integrated into a larger Streamlit application.
Your friend can import and call render_review_clustering_page() from their main app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Custom CSS for styling
def inject_custom_css():
    """Add custom CSS styling to the page"""
    st.markdown("""
        <style>
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .insight-box {
            background-color: #f0f8ff;
            border-left: 5px solid #3498db;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_review_data():
    """Load the processed review data with clustering results"""
    try:
        df = pd.read_csv('data/streamlit_export/positive_reviews_with_clusters.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_aspect_profiles():
    """Load aspect profiles by cluster"""
    try:
        df = pd.read_csv('data/streamlit_export/aspect_cluster_profiles.csv', index_col=0)
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_genre_profiles():
    """Load aspect profiles by genre"""
    try:
        df = pd.read_csv('data/streamlit_export/genre_aspect_profiles.csv', index_col=0)
        return df
    except FileNotFoundError:
        return None

# Visualization functions
def create_wordcloud_grid(df, cluster_col='cluster', text_col='tokens_text'):
    """Generate word clouds for each cluster"""
    n_clusters = df[cluster_col].nunique()
    
    cluster_names = {
        0: "Sci-Fi/Space Films (Interstellar, Nolan)",
        1: "General Positive Reviews",
        2: "Fantasy/Magic (Harry Potter)",
        3: "Quality Films & Storytelling",
        4: "Drama/Prison (Shawshank Redemption)"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Overall word cloud
    all_text = ' '.join(df[text_col].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         colormap='Greens', max_words=100).generate(all_text)
    axes[0].imshow(wordcloud, interpolation='bilinear')
    axes[0].set_title(f'All Positive Reviews\n(n={len(df):,})', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Individual cluster word clouds
    for i in range(min(n_clusters, 5)):
        cluster_text = ' '.join(df[df[cluster_col]==i][text_col].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='viridis', max_words=80).generate(cluster_text)
        axes[i+1].imshow(wordcloud, interpolation='bilinear')
        axes[i+1].set_title(f'Cluster {i}: {cluster_names.get(i, "Unnamed")}\n(n={len(df[df[cluster_col]==i]):,})', 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig

def create_tsne_visualization(df, tsne_x='tsne_x', tsne_y='tsne_y', cluster_col='cluster'):
    """Create t-SNE visualization of clusters"""
    n_clusters = df[cluster_col].nunique()
    
    cluster_names = {
        0: "Sci-Fi/Space Films",
        1: "General Positive",
        2: "Fantasy/Magic",
        3: "Quality & Storytelling",
        4: "Drama/Character"
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for i in range(n_clusters):
        cluster_data = df[df[cluster_col] == i]
        ax.scatter(cluster_data[tsne_x], cluster_data[tsne_y], 
                  c=colors[i], 
                  label=f'Cluster {i}: {cluster_names.get(i, "Unnamed")} (n={len(cluster_data):,})',
                  alpha=0.7, s=40, edgecolors='white', linewidth=0.8)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Positive Review Clusters Visualized with t-SNE', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_aspect_heatmap(cluster_profiles):
    """Create heatmap of aspect profiles by cluster"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(cluster_profiles.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Aspect Score'},
                linewidths=1, linecolor='white', ax=ax)
    ax.set_title('Aspect Profiles by Cluster\n(Shows what each cluster values most)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Aspect', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_genre_heatmap(genre_profiles):
    """Create heatmap of aspect preferences by genre"""
    if 'Total' in genre_profiles.columns:
        genre_profiles = genre_profiles.drop('Total', axis=1)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(genre_profiles, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=0, vmax=1.0,
                cbar_kws={'label': 'Average Aspect Score'},
                linewidths=1, linecolor='white', ax=ax)
    ax.set_title('Aspect Preferences by Genre\n(What viewers value in positive reviews)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Aspect', fontsize=12, fontweight='bold')
    ax.set_ylabel('Genre', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig

def create_genre_comparison(genre_profiles, selected_genres):
    """Create bar chart comparing selected genres"""
    if 'Total' in genre_profiles.columns:
        genre_data = genre_profiles.loc[selected_genres].drop('Total', axis=1)
    else:
        genre_data = genre_profiles.loc[selected_genres]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    genre_data.T.plot(kind='bar', ax=ax, width=0.75, colormap='Set2')
    ax.set_title('Aspect Preferences Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Aspect', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=11, fontweight='bold')
    ax.legend(title='Genre', fontsize=9, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig

# Main page function - this is what your friend will call
def render_review_clustering_page():
    """
    Main function to render the Review Clustering Analysis page.
    
    Call this function from your multi-page Streamlit app to display
    the review clustering analysis page.
    """
    
    # Inject custom CSS
    inject_custom_css()
    
    # Page title
    st.title("Movie Review Clustering Analysis")
    
    st.markdown("""
    <div class="insight-box">
    <p><strong>Understanding What Viewers Value in Movies</strong></p>
    <p>This analysis explores positive movie reviews to uncover patterns in what audiences appreciate. 
    Through clustering and aspect-based analysis, we identify the key factors that drive positive 
    sentiment: plot quality, acting performance, visual appeal, emotional impact, and more.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_review_data()
        aspect_profiles = load_aspect_profiles()
        genre_profiles = load_genre_profiles()
    
    if df is None:
        st.error("""
        ### Data Not Found
        
        Please run the analysis notebook (`07_review_text_analysis.ipynb`) to generate the required data files.
        
        Expected files in `data/streamlit_export/`:
        - `positive_reviews_with_clusters.csv`
        - `aspect_cluster_profiles.csv`
        - `genre_aspect_profiles.csv`
        """)
        return
    
    # Overview metrics
    st.header("Analysis Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Positive Reviews Analyzed", f"{len(df):,}")
    with col2:
        if 'movie_id' in df.columns:
            st.metric("Unique Movies", f"{df['movie_id'].nunique():,}")
    with col3:
        if 'cluster' in df.columns:
            st.metric("Review Clusters", df['cluster'].nunique())
    
    # Section selector
    st.markdown("---")
    section = st.radio(
        "**Choose a visualization:**",
        ["Word Cloud Analysis", "Cluster Visualization (t-SNE)", 
         "Aspect-Based Analysis", "Genre Insights"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Display selected section
    if section == "Word Cloud Analysis":
        st.header("Word Cloud Analysis")
        
        st.markdown("""
        **Word clouds visualize the most frequently mentioned terms in each cluster.**
        
        Larger words appear more frequently in reviews. Each cluster represents a group of reviews 
        that share similar vocabulary and themes. Notice how different clusters emphasize different 
        aspects: some focus on specific movies (Interstellar, Harry Potter), while others highlight 
        general film qualities (story, characters, emotions).
        """)
        
        if 'tokens_text' in df.columns and 'cluster' in df.columns:
            with st.spinner('Generating word clouds...'):
                fig = create_wordcloud_grid(df)
                st.pyplot(fig)
            
            with st.expander("Cluster Interpretations"):
                st.markdown("""
                - **Cluster 0 (Sci-Fi/Space)**: Reviews emphasize scientific concepts, space themes, 
                  and directors like Christopher Nolan
                - **Cluster 1 (General Positive)**: Broad appreciation across multiple aspects without 
                  specific focus
                - **Cluster 2 (Fantasy/Magic)**: Fantasy elements, magical themes, book adaptations
                - **Cluster 3 (Quality & Storytelling)**: Focus on narrative structure, script quality, 
                  and filmmaking craft
                - **Cluster 4 (Drama/Character)**: Emphasis on emotional depth, character development, 
                  and performances
                """)
        else:
            st.warning("Required columns not found in dataset.")
    
    elif section == "Cluster Visualization (t-SNE)":
        st.header("Cluster Visualization (t-SNE)")
        
        st.markdown("""
        **t-SNE (t-Distributed Stochastic Neighbor Embedding) projects high-dimensional 
        review data into 2D space.**
        
        Points that are close together represent reviews with similar content and vocabulary. 
        Each color represents a different cluster. Notice how some clusters are tightly grouped 
        (indicating very similar reviews) while others are more spread out (indicating more 
        diversity within the cluster).
        """)
        
        if all(col in df.columns for col in ['tsne_x', 'tsne_y', 'cluster']):
            fig = create_tsne_visualization(df)
            st.pyplot(fig)
            
            with st.expander("What This Shows"):
                st.markdown("""
                - **Clear Separation**: Some clusters (like Sci-Fi and Drama) are well-separated, 
                  indicating distinct vocabulary patterns
                - **Overlap Regions**: General positive clusters show more overlap, suggesting 
                  these represent more general sentiments
                - **Cluster Density**: Tighter clusters indicate reviews that are very similar to each 
                  other, often focused on specific movies or themes
                """)
        else:
            st.warning("t-SNE coordinates not found. Please regenerate the data with t-SNE coordinates.")
    
    elif section == "Aspect-Based Analysis":
        st.header("Aspect-Based Analysis")
        
        st.markdown("""
        **Aspect analysis measures how much each cluster emphasizes different movie qualities.**
        
        We identified 10 key aspects that viewers mention in positive reviews: plot/story, 
        acting, visuals, emotions, action, music, direction, originality, entertainment, and pacing. 
        The heatmap shows which aspects each cluster values most.
        """)
        
        if aspect_profiles is not None:
            fig = create_aspect_heatmap(aspect_profiles)
            st.pyplot(fig)
            
            with st.expander("Key Insights"):
                st.markdown("""
                The aspect profile heatmap reveals distinct viewing preferences:
                
                - **Cluster 0 - Entertainment-Focused** (n=4,627): Values fun and enjoyment above all
                
                - **Cluster 1 - Music & Sound Enthusiasts** (n=2,379): Exceptionally high music/sound scores (1.93)
                
                - **Cluster 2 - Casual/Nostalgic Viewers** (n=10,181): Balanced but lower intensity appreciation
                
                - **Cluster 3 - Craft & Direction Appreciators** (n=3,193): Focuses on directorial vision and filmmaking
                
                - **Cluster 4 - Story & Performance Lovers** (n=4,358): Values strong narratives and acting
                
                - **Cluster 5 - Passionate Cinephiles** (n=1,482): Highest scores across ALL aspects
                
                These patterns reveal **why** viewers appreciated these films, not just that they liked them.
                """)
            
            # Cluster size distribution
            if 'aspect_cluster' in df.columns:
                st.subheader("Cluster Sizes")
                cluster_counts = df['aspect_cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)
        else:
            st.warning("Aspect profile data not found.")
    
    elif section == "Genre Insights":
        st.header("Genre Insights")
        
        st.markdown("""
        **Different genres show distinct patterns in what viewers value.**
        
        By analyzing aspect scores across genres, we can see what matters most to fans of 
        each genre. For example, action fans emphasize excitement and action sequences, while 
        drama fans focus more on emotional impact and acting performances.
        """)
        
        if genre_profiles is not None:
            # Full heatmap
            st.subheader("Complete Genre-Aspect Heatmap")
            fig = create_genre_heatmap(genre_profiles)
            st.pyplot(fig)
            
            with st.expander("Genre-Specific Preferences"):
                st.markdown("""
                The heatmap reveals clear genre preferences:
                
                - **Adventure**: High scores for plot/story and acting - fans value narrative quality
                - **Action**: Emphasis on action/excitement (naturally), but also values acting
                - **Horror**: Unique focus on originality and atmosphere over other aspects
                - **Drama**: Strong emotional impact and acting performance scores
                - **Comedy**: Relatively balanced, with focus on entertainment value
                """)
            
            # Interactive genre comparison
            st.subheader("Compare Specific Genres")
            st.markdown("Select genres to compare their aspect preferences side-by-side:")
            
            available_genres = genre_profiles.index.tolist()
            selected_genres = st.multiselect(
                "Select genres:",
                available_genres,
                default=available_genres[:3] if len(available_genres) >= 3 else available_genres
            )
            
            if len(selected_genres) > 0:
                fig = create_genre_comparison(genre_profiles, selected_genres)
                st.pyplot(fig)
                
                st.info("Use the selector above to compare different genres and identify what makes each genre's audience unique.")
        else:
            st.warning("Genre profile data not found.")

# For standalone testing - your friend won't need this
if __name__ == "__main__":
    st.set_page_config(
        page_title="Review Clustering Analysis",
        page_icon="📊",
        layout="wide"
    )
    render_review_clustering_page()
