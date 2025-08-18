import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agent-Based Humor Generation",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .humor-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background-color: #f9f9f9;
    }
    .persona-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .score-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .score-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .score-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to the backend"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return {}

def get_score_color_class(score: float) -> str:
    """Get CSS class based on score"""
    if score >= 7.0:
        return "score-high"
    elif score >= 5.0:
        return "score-medium"
    else:
        return "score-low"

def format_score(score: float) -> str:
    """Format score with color"""
    css_class = get_score_color_class(score)
    return f'<span class="{css_class}">{score:.1f}</span>'

# Sidebar
st.sidebar.title("🎭 Agent Configuration")

# User selection/creation
st.sidebar.subheader("👤 User Management")

# Check if we have users in session state
if 'users' not in st.session_state:
    st.session_state.users = []

# Get users
users_data = make_api_request("/users/1")  # Try to get a user to check API
if users_data:
    st.session_state.current_user_id = 1
else:
    st.session_state.current_user_id = None

# User creation form
with st.sidebar.expander("Create New User"):
    username = st.text_input("Username")
    email = st.text_input("Email")
    age_range = st.selectbox("Age Range", 
                            ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    occupation = st.text_input("Occupation")
    education = st.selectbox("Education Level", 
                            ["High School", "College", "Graduate", "PhD"])
    interests = st.multiselect("Interests", 
                              ["Technology", "Sports", "Arts", "Science", "Politics", 
                               "Entertainment", "Travel", "Food", "Music", "Literature"])
    
    if st.button("Create User"):
        user_data = {
            "username": username,
            "email": email,
            "age_range": age_range,
            "occupation": occupation,
            "education_level": education,
            "interests": interests
        }
        result = make_api_request("/users", "POST", user_data)
        if result:
            st.success(f"User created! ID: {result.get('user_id')}")
            st.session_state.current_user_id = result.get('user_id')

# Generation parameters
st.sidebar.subheader("⚙️ Generation Settings")
num_generators = st.sidebar.slider("Number of Generator Agents", 1, 5, 3)
num_evaluators = st.sidebar.slider("Number of Evaluator Agents", 1, 3, 2)

# Main content
st.title("🤖 Agent-Based Humor Generation System")
st.markdown("Generate personalized humor using multiple AI agents with different personas")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎭 Generate Humor", "📊 Analytics", "👥 Personas", "💾 History"])

with tab1:
    st.header("Generate Humor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        with st.form("humor_generation_form"):
            context = st.text_area(
                "Context/Setting", 
                placeholder="e.g., Office meeting, birthday party, casual conversation with friends...",
                height=100
            )
            
            audience = st.text_input(
                "Target Audience", 
                placeholder="e.g., colleagues, family, young adults, professionals..."
            )
            
            topic = st.text_input(
                "Topic/Subject", 
                placeholder="e.g., work stress, technology, relationships, current events..."
            )
            
            humor_type = st.selectbox(
                "Humor Type (Optional)",
                ["", "witty", "sarcastic", "pun", "observational", "dad joke", "wordplay"]
            )
            
            submitted = st.form_submit_button("🎭 Generate Humor", use_container_width=True)
    
    with col2:
        # Current user info
        if st.session_state.current_user_id:
            user_data = make_api_request(f"/users/{st.session_state.current_user_id}")
            if user_data:
                st.info(f"**Current User:** {user_data.get('username', 'Unknown')}")
                st.caption(f"Age: {user_data.get('age_range', 'N/A')}")
                st.caption(f"Occupation: {user_data.get('occupation', 'N/A')}")
        
        # Quick examples
        st.subheader("💡 Quick Examples")
        if st.button("Office Humor"):
            st.session_state.example_context = "Office meeting"
            st.session_state.example_audience = "colleagues"
            st.session_state.example_topic = "deadline pressure"
        
        if st.button("Party Jokes"):
            st.session_state.example_context = "Birthday party"
            st.session_state.example_audience = "friends"
            st.session_state.example_topic = "getting older"
        
        if st.button("Tech Humor"):
            st.session_state.example_context = "Tech conference"
            st.session_state.example_audience = "developers"
            st.session_state.example_topic = "coding bugs"
    
    # Generate humor
    if submitted and context and audience and topic:
        with st.spinner("🤖 Agents are generating humor..."):
            generation_data = {
                "context": context,
                "audience": audience,
                "topic": topic,
                "user_id": st.session_state.current_user_id,
                "humor_type": humor_type if humor_type else None,
                "num_generators": num_generators,
                "num_evaluators": num_evaluators
            }
            
            result = make_api_request("/generate-humor", "POST", generation_data)
            
            if result and result.get('success'):
                st.success(f"✅ Generated {result['total_generations']} humor options!")
                
                # Display generation info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Generations", result['total_generations'])
                with col2:
                    st.metric("Generator Personas", len(result['generation_personas']))
                with col3:
                    st.metric("Evaluator Personas", len(result['evaluation_personas']))
                
                # Display personas used
                st.subheader("🎭 Personas Used")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Generators:**")
                    for persona in result['generation_personas']:
                        st.markdown(f"• {persona}")
                
                with col2:
                    st.write("**Evaluators:**")
                    for persona in result['evaluation_personas']:
                        st.markdown(f"• {persona}")
                
                # Display results
                st.subheader("🏆 Top Humor Results")
                
                for i, humor_result in enumerate(result['top_results'], 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="humor-card">
                            <h4>#{i} - <span class="persona-badge">{humor_result['persona_name']}</span></h4>
                            <p style="font-size: 16px; font-style: italic;">"{humor_result['text']}"</p>
                            <hr>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Humor: {format_score(humor_result['humor_score'])}</span>
                                <span>Creativity: {format_score(humor_result['creativity_score'])}</span>
                                <span>Appropriateness: {format_score(humor_result['appropriateness_score'])}</span>
                                <span>Relevance: {format_score(humor_result['context_relevance_score'])}</span>
                                <span><strong>Overall: {format_score(humor_result['overall_score'])}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feedback buttons
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                        with col1:
                            if st.button("👍", key=f"like_{i}"):
                                # Submit positive feedback
                                feedback_data = {
                                    "generation_id": humor_result['id'],
                                    "liked": True,
                                    "humor_rating": 5
                                }
                                make_api_request("/feedback", "POST", feedback_data)
                                st.success("Thanks for the feedback!")
                        
                        with col2:
                            if st.button("👎", key=f"dislike_{i}"):
                                # Submit negative feedback
                                feedback_data = {
                                    "generation_id": humor_result['id'],
                                    "liked": False,
                                    "humor_rating": 1
                                }
                                make_api_request("/feedback", "POST", feedback_data)
                                st.success("Thanks for the feedback!")
                        
                        with col3:
                            rating = st.selectbox("Rate", [1, 2, 3, 4, 5], 
                                                index=2, key=f"rating_{i}")
                        
                        # Show detailed evaluations
                        with st.expander(f"View Detailed Evaluations for #{i}"):
                            for eval_data in humor_result['evaluations']:
                                st.write(f"**{eval_data['evaluator_name']}** ({eval_data['model_used']}):")
                                scores = eval_data['scores']
                                st.write(f"Scores: H:{scores['humor_score']:.1f} | "
                                       f"C:{scores['creativity_score']:.1f} | "
                                       f"A:{scores['appropriateness_score']:.1f} | "
                                       f"R:{scores['context_relevance_score']:.1f}")
                                st.write(f"*{eval_data['reasoning'][:200]}...*")
                                st.write("---")

with tab2:
    st.header("📊 Analytics Dashboard")
    
    # User analytics
    if st.session_state.current_user_id:
        user_analytics = make_api_request(f"/analytics/user/{st.session_state.current_user_id}")
        
        if user_analytics:
            st.subheader(f"👤 Your Analytics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Generations", user_analytics['total_generations'])
            with col2:
                st.metric("Total Feedback", user_analytics['total_feedback'])
            with col3:
                rate = user_analytics['positive_feedback_rate'] * 100
                st.metric("Positive Feedback", f"{rate:.1f}%")
            
            # Top personas for user
            if user_analytics['top_personas']:
                st.subheader("🎭 Your Favorite Personas")
                for persona in user_analytics['top_personas']:
                    st.write(f"• **{persona['name']}**: {persona['description']}")
    
    # Overall persona analytics
    persona_analytics = make_api_request("/analytics/personas")
    
    if persona_analytics:
        st.subheader("🏆 Persona Performance")
        
        personas_df = pd.DataFrame(persona_analytics['personas'])
        
        if not personas_df.empty:
            # Performance chart
            fig = px.scatter(
                personas_df, 
                x='total_generations', 
                y='average_rating',
                size='total_generations',
                hover_name='name',
                title="Persona Performance: Usage vs Rating"
            )
            fig.update_layout(
                xaxis_title="Total Generations",
                yaxis_title="Average Rating"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers table
            st.subheader("📈 Top Performing Personas")
            top_personas = personas_df.head(5)
            st.dataframe(
                top_personas[['name', 'total_generations', 'average_rating', 'expertise_areas']],
                use_container_width=True
            )

with tab3:
    st.header("👥 Persona Gallery")
    
    personas = make_api_request("/personas")
    
    if personas:
        # Search and filter
        search_term = st.text_input("🔍 Search personas...")
        
        filtered_personas = personas
        if search_term:
            filtered_personas = [
                p for p in personas 
                if search_term.lower() in p['name'].lower() or 
                   search_term.lower() in p['description'].lower()
            ]
        
        # Display personas in cards
        for i in range(0, len(filtered_personas), 2):
            col1, col2 = st.columns(2)
            
            for j, col in enumerate([col1, col2]):
                if i + j < len(filtered_personas):
                    persona = filtered_personas[i + j]
                    
                    with col:
                        with st.container():
                            st.markdown(f"""
                            <div class="humor-card">
                                <h3>{persona['name']}</h3>
                                <p>{persona['description']}</p>
                                <p><strong>Expertise:</strong> {', '.join(persona['expertise_areas'])}</p>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Generations: {persona['total_generations']}</span>
                                    <span>Rating: {persona['avg_rating']:.1f}/10</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show details
                            with st.expander("View Details"):
                                st.write("**Demographics:**")
                                for key, value in persona['demographics'].items():
                                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
                                
                                st.write("**Personality Traits:**")
                                for key, value in persona['personality_traits'].items():
                                    st.write(f"• {key.replace('_', ' ').title()}: {value}")

with tab4:
    st.header("💾 Generation History")
    
    if st.session_state.current_user_id:
        st.info("Generation history will be displayed here once implemented in the backend.")
        
        # Placeholder for future implementation
        st.subheader("Recent Generations")
        st.write("This feature will show your recent humor generations with their scores and feedback.")
        
        st.subheader("Favorite Topics")
        st.write("This will show topics you've generated humor about most frequently.")
        
        st.subheader("Performance Over Time")
        st.write("This will show how your humor preferences have evolved over time.")
    else:
        st.warning("Please create a user account to view generation history.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        🤖 Agent-Based Humor Generation System | 
        Powered by Multiple LLMs with Persona-Based Agents
    </div>
    """, 
    unsafe_allow_html=True
) 