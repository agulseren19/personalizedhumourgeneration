#!/usr/bin/env python3
"""
Feedback UI for Humor Generation System
Streamlit app for user interaction and feedback collection
"""

import streamlit as st
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.humor_agents import HumorAgentOrchestrator, HumorRequest
from personas.persona_manager import PersonaManager
from knowledge.aws_knowledge_base import aws_knowledge_base, UserPreference
from llm_clients.multi_llm_manager import multi_llm_manager
from models.database import get_session_local, create_database
from config.settings import settings

# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize the humor generation system"""
    try:
        # Database setup
        create_database(settings.database_url)
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        # Initialize managers
        persona_manager = PersonaManager(db)
        orchestrator = HumorAgentOrchestrator(persona_manager)
        
        return orchestrator, persona_manager, db
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="🎭 AI Humor Generator",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system
    orchestrator, persona_manager, db = initialize_system()
    
    if not orchestrator:
        st.error("❌ Failed to initialize system")
        return
    
    # Title and description
    st.title("🎭 AI Humor Generation System")
    st.markdown("""
    **Multi-Agent Humor Generator** with personalized learning!
    
    This system uses multiple AI personas (GPT-4, Claude, DeepSeek) to generate humor
    tailored to your preferences. Your feedback helps the system learn what you like!
    """)
    
    # Sidebar for user settings
    with st.sidebar:
        st.header("🎯 Settings")
        
        # User ID
        user_id = st.text_input("👤 User ID", value="demo_user", help="Enter your unique user ID")
        
        # Context and audience
        st.subheader("📝 Generation Settings")
        context = st.text_input(
            "🎯 Context", 
            value="What did I bring back from vacation? _____",
            help="The setup or black card for humor generation"
        )
        
        audience = st.selectbox(
            "👥 Audience",
            ["adults", "family", "colleagues", "friends", "general"],
            index=0,
            help="Who is the target audience?"
        )
        
        topic = st.text_input(
            "🏷️ Topic",
            value="travel experiences",
            help="The general topic or theme"
        )
        
        # Number of agents
        num_agents = st.slider(
            "🤖 Number of Agents",
            min_value=1,
            max_value=5,
            value=3,
            help="How many different AI personas to use"
        )
        
        # Show available LLMs
        st.subheader("🔧 Available Models")
        available_models = multi_llm_manager.get_available_models()
        for model in available_models:
            st.info(f"✅ {model.value}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 Generate Humor", 
        "📊 Your Preferences", 
        "👥 Group Mode",
        "📈 Analytics"
    ])
    
    with tab1:
        generate_humor_tab(orchestrator, user_id, context, audience, topic, num_agents)
    
    with tab2:
        user_preferences_tab(user_id)
    
    with tab3:
        group_mode_tab(orchestrator)
    
    with tab4:
        analytics_tab(user_id)

def generate_humor_tab(orchestrator, user_id, context, audience, topic, num_agents):
    """Main humor generation interface"""
    st.header("🎭 Generate Humor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Humor", type="primary", use_container_width=True):
            with st.spinner("🤖 AI agents are working on your humor..."):
                # Create humor request
                request = HumorRequest(
                    context=context,
                    audience=audience,
                    topic=topic,
                    user_id=user_id
                )
                
                # Generate humor
                try:
                    result = asyncio.run(
                        orchestrator.generate_and_evaluate_humor(
                            request, 
                            num_generators=num_agents,
                            num_evaluators=1
                        )
                    )
                    
                    if result['success']:
                        st.success(f"✅ Generated {len(result['top_results'])} humor options!")
                        
                        # Display results
                        for i, ranked_result in enumerate(result['top_results'], 1):
                            generation = ranked_result['generation']
                            scores = ranked_result['average_scores']
                            
                            # Create card for each result
                            with st.container():
                                st.markdown(f"### Option {i} - {generation.persona_name}")
                                
                                # Show the humor
                                st.markdown(f"""
                                **🎯 Context**: {context}
                                
                                **😂 Response**: "{generation.text}"
                                
                                **🤖 Generated by**: {generation.persona_name} using {generation.model_used}
                                """)
                                
                                # Show scores
                                col_score1, col_score2, col_score3 = st.columns(3)
                                with col_score1:
                                    st.metric("😂 Humor Score", f"{scores['humor_score']:.1f}/10")
                                with col_score2:
                                    st.metric("🎨 Creativity", f"{scores['creativity_score']:.1f}/10")
                                with col_score3:
                                    st.metric("✅ Overall", f"{scores['overall_score']:.1f}/10")
                                
                                # Feedback section
                                st.markdown("#### 📝 Your Feedback")
                                col_feedback, col_submit = st.columns([3, 1])
                                
                                with col_feedback:
                                    feedback_score = st.slider(
                                        f"Rate Option {i}",
                                        min_value=1,
                                        max_value=10,
                                        value=5,
                                        key=f"feedback_{i}",
                                        help="1 = Terrible, 10 = Hilarious!"
                                    )
                                
                                with col_submit:
                                    if st.button(f"Submit", key=f"submit_{i}"):
                                        # Store feedback
                                        asyncio.run(
                                            aws_knowledge_base.update_user_feedback(
                                                user_id=user_id,
                                                persona_name=generation.persona_name,
                                                feedback_score=feedback_score,
                                                context=context
                                            )
                                        )
                                        st.success("✅ Feedback saved!")
                                        st.experimental_rerun()
                                
                                st.divider()
                    
                    else:
                        st.error(f"❌ Generation failed: {result.get('error')}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.markdown("### 🎯 Quick Tips")
        st.info("""
        **💡 For better results:**
        - Be specific with context
        - Choose appropriate audience
        - Provide feedback to improve future generations
        - Try different topics and contexts
        """)
        
        st.markdown("### 🤖 Current Personas")
        if 'orchestrator' in locals():
            try:
                # Get recommended personas for current context
                recommended = asyncio.run(
                    aws_knowledge_base.get_persona_recommendations(
                        user_id=user_id,
                        context=context,
                        audience=audience
                    )
                )
                
                st.markdown("**Recommended for you:**")
                for persona in recommended:
                    st.markdown(f"• {persona}")
                    
            except Exception as e:
                st.warning("Could not load persona recommendations")

def user_preferences_tab(user_id):
    """Display and manage user preferences"""
    st.header("📊 Your Humor Preferences")
    
    try:
        # Get user preferences
        user_pref = asyncio.run(aws_knowledge_base.get_user_preference(user_id))
        
        if user_pref:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👍 Liked Personas")
                if user_pref.liked_personas:
                    for persona in user_pref.liked_personas:
                        st.success(f"✅ {persona}")
                else:
                    st.info("No liked personas yet. Give some feedback!")
                
                st.subheader("📈 Context Preferences")
                if user_pref.context_preferences:
                    # Create bar chart of context preferences
                    contexts = list(user_pref.context_preferences.keys())
                    scores = list(user_pref.context_preferences.values())
                    
                    fig = px.bar(
                        x=contexts,
                        y=scores,
                        title="Context Performance",
                        labels={'x': 'Context', 'y': 'Preference Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No context data yet.")
            
            with col2:
                st.subheader("👎 Disliked Personas")
                if user_pref.disliked_personas:
                    for persona in user_pref.disliked_personas:
                        st.error(f"❌ {persona}")
                else:
                    st.info("No disliked personas yet.")
                
                st.subheader("📊 Interaction History")
                if user_pref.interaction_history:
                    # Show recent interactions
                    recent_interactions = user_pref.interaction_history[-10:]
                    
                    df = pd.DataFrame(recent_interactions)
                    if not df.empty:
                        fig = px.line(
                            df,
                            x=range(len(df)),
                            y='feedback_score',
                            title="Recent Feedback Scores",
                            labels={'x': 'Interaction', 'y': 'Score'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No interaction history yet.")
            
            # Profile summary
            st.subheader("📋 Profile Summary")
            st.json({
                "User ID": user_pref.user_id,
                "Total Interactions": len(user_pref.interaction_history),
                "Liked Personas": len(user_pref.liked_personas),
                "Disliked Personas": len(user_pref.disliked_personas),
                "Last Updated": user_pref.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        else:
            st.info("🆕 No preferences found. Start generating humor to build your profile!")
            
            # Option to create demo preferences
            if st.button("🎯 Create Demo Profile"):
                demo_pref = UserPreference(
                    user_id=user_id,
                    humor_styles=["witty", "clever"],
                    liked_personas=["dad_humor_enthusiast"],
                    disliked_personas=[],
                    context_preferences={},
                    demographic_profile={"age_range": "25-35"},
                    interaction_history=[],
                    last_updated=datetime.now()
                )
                
                asyncio.run(aws_knowledge_base.store_user_preference(demo_pref))
                st.success("✅ Demo profile created!")
                st.experimental_rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading preferences: {e}")

def group_mode_tab(orchestrator):
    """Group humor generation interface"""
    st.header("👥 Group Humor Mode")
    st.markdown("Generate humor that appeals to multiple users!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("👥 Group Setup")
        
        # Group ID
        group_id = st.text_input("🏷️ Group ID", value="demo_group")
        
        # Member IDs
        member_input = st.text_area(
            "👤 Member IDs (one per line)",
            value="demo_user\nuser2\nuser3",
            help="Enter user IDs, one per line"
        )
        member_ids = [uid.strip() for uid in member_input.split('\n') if uid.strip()]
        
        st.info(f"Group has {len(member_ids)} members: {', '.join(member_ids)}")
        
        # Group context
        group_context_input = st.text_input(
            "🎯 Group Context",
            value="What's the best team building activity? _____"
        )
        
        if st.button("🚀 Generate Group Humor", type="primary"):
            with st.spinner("🤖 Creating group context and generating humor..."):
                try:
                    # Create group context
                    group_context = asyncio.run(
                        aws_knowledge_base.create_group_context(group_id, member_ids)
                    )
                    
                    st.success(f"✅ Group context created!")
                    st.json({
                        "Common Humor Styles": group_context.common_humor_styles,
                        "Member Count": len(group_context.member_ids)
                    })
                    
                    # Generate humor for group
                    request = HumorRequest(
                        context=group_context_input,
                        audience="group",
                        topic="team activity",
                        user_id=None  # Group mode
                    )
                    
                    result = asyncio.run(
                        orchestrator.generate_and_evaluate_humor(request, num_generators=3)
                    )
                    
                    if result['success']:
                        st.success("🎭 Group humor generated!")
                        
                        for i, ranked_result in enumerate(result['top_results'], 1):
                            generation = ranked_result['generation']
                            scores = ranked_result['average_scores']
                            
                            with st.container():
                                st.markdown(f"**Option {i}**: {generation.text}")
                                st.caption(f"By {generation.persona_name} • Score: {scores['overall_score']:.1f}/10")
                                st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Group humor generation failed: {e}")
    
    with col2:
        st.markdown("### 💡 Group Mode Tips")
        st.info("""
        **Group mode features:**
        - Analyzes all members' preferences
        - Finds common humor styles
        - Avoids personas disliked by members
        - Creates inclusive humor
        """)

def analytics_tab(user_id):
    """Analytics and insights dashboard"""
    st.header("📈 Analytics Dashboard")
    
    try:
        # Get user preferences for analytics
        user_pref = asyncio.run(aws_knowledge_base.get_user_preference(user_id))
        
        if user_pref and user_pref.interaction_history:
            # Performance over time
            df = pd.DataFrame(user_pref.interaction_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Feedback Trends")
                fig = px.line(
                    df.tail(20),  # Last 20 interactions
                    x='timestamp',
                    y='feedback_score',
                    title="Your Feedback Over Time",
                    labels={'feedback_score': 'Score', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Average scores by persona
                persona_scores = df.groupby('persona_name')['feedback_score'].mean().sort_values(ascending=False)
                
                fig_persona = px.bar(
                    x=persona_scores.index,
                    y=persona_scores.values,
                    title="Average Score by Persona",
                    labels={'x': 'Persona', 'y': 'Average Score'}
                )
                st.plotly_chart(fig_persona, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Context Analysis")
                context_scores = df.groupby('context')['feedback_score'].agg(['mean', 'count'])
                context_scores = context_scores[context_scores['count'] >= 2]  # At least 2 interactions
                
                if not context_scores.empty:
                    fig_context = px.scatter(
                        x=context_scores['count'],
                        y=context_scores['mean'],
                        title="Context Performance vs Frequency",
                        labels={'x': 'Number of Interactions', 'y': 'Average Score'},
                        hover_data={'context': context_scores.index}
                    )
                    st.plotly_chart(fig_context, use_container_width=True)
                
                # Summary stats
                st.subheader("📋 Summary Statistics")
                st.metric("Total Interactions", len(df))
                st.metric("Average Score", f"{df['feedback_score'].mean():.1f}")
                st.metric("Best Performing Persona", persona_scores.index[0] if not persona_scores.empty else "None")
                st.metric("Improvement Trend", "📈 Improving" if df.tail(5)['feedback_score'].mean() > df.head(5)['feedback_score'].mean() else "📉 Declining")
        
        else:
            st.info("🆕 Not enough data for analytics yet. Generate more humor to see insights!")
    
    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")

if __name__ == "__main__":
    main() 