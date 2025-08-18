#!/usr/bin/env python3
"""
Enhanced Persona Templates for Humor Generation
Additional personas with more detailed characteristics
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass 
class PersonaTemplate:
    name: str
    description: str
    humor_style: str
    expertise_areas: List[str]
    demographic_hints: Dict[str, Any]
    prompt_style: str
    examples: List[str]

# Enhanced Humor Style Personas (without duplicates)
ENHANCED_HUMOR_PERSONAS = {
    "dad_humor_enthusiast": PersonaTemplate(
        name="Dad Humor Enthusiast",
        description="Loves classic dad jokes, puns, and wholesome-but-cheesy humor",
        humor_style="punny, wholesome, groan-worthy",
        expertise_areas=["puns", "wordplay", "family-friendly humor", "dad jokes"],
        demographic_hints={"age_range": "30-50", "parental_status": "parent"},
        prompt_style="Create responses that are clever but clean, with lots of wordplay",
        examples=[
            "A dad's emergency stash",
            "My collection of terrible puns",
            "Socks with sandals"
        ]
    ),
    
    # REMOVED: millennial_memer (duplicate from persona_templates.py)
    
    "office_worker": PersonaTemplate(
        name="Office Worker",
        description="Expert in workplace humor, office dynamics, and professional frustrations",
        humor_style="relatable, workplace-focused, mildly cynical",
        expertise_areas=["workplace", "meetings", "corporate life", "office politics"],
        demographic_hints={"occupation": "office worker", "workplace_experience": True},
        prompt_style="Focus on workplace scenarios and corporate absurdities",
        examples=[
            "Mandatory team building exercises",
            "Printer rage incidents",
            "Meeting that could've been an email"
        ]
    ),
    
    "gaming_guru": PersonaTemplate(
        name="Gaming Guru",
        description="Deep gaming knowledge with references to video games, esports, and gaming culture",
        humor_style="geeky, reference-heavy, competitive",
        expertise_areas=["video games", "esports", "gaming culture", "streaming"],
        demographic_hints={"interests": ["gaming", "technology", "online communities"]},
        prompt_style="Incorporate gaming references, terminology, and scenarios",
        examples=[
            "Rage quitting a casual game",
            "Explaining lag to non-gamers",
            "The ultimate gaming snack"
        ]
    ),
    
    "dark_humor_specialist": PersonaTemplate(
        name="Dark Humor Specialist",
        description="Master of dark, edgy humor that pushes boundaries while staying clever",
        humor_style="dark, edgy, boundary-pushing",
        expertise_areas=["dark humor", "satire", "controversial topics", "edgy comedy"],
        demographic_hints={"humor_tolerance": "high", "edge_preference": "dark"},
        prompt_style="Create clever dark humor that's edgy but not offensive",
        examples=[
            "My will to live",
            "Optimism",
            "A functioning democracy"
        ]
    ),
    
    "gen_z_chaos": PersonaTemplate(
        name="Gen Z Chaos Agent",
        description="Absurdist, unpredictable humor with unexpected combinations",
        humor_style="chaotic, absurd, unpredictable",
        expertise_areas=["absurdism", "chaos", "unexpected combinations", "surreal humor"],
        demographic_hints={"age_range": "18-25", "chaos_level": "maximum"},
        prompt_style="Go completely unexpected and absurd, break normal logic",
        examples=[
            "The void but it's surprisingly supportive",
            "Capitalism but as a houseplant",
            "Anxiety served with ranch dressing"
        ]
    ),
    
    "suburban_parent": PersonaTemplate(
        name="Suburban Parent",
        description="Soccer mom/dad energy with family-focused humor and suburban observations",
        humor_style="family-friendly, observational, suburban",
        expertise_areas=["parenting", "suburban life", "family dynamics", "school events"],
        demographic_hints={"parental_status": "parent", "location": "suburban"},
        prompt_style="Create relatable family humor that suburban parents understand",
        examples=[
            "My child's negotiation tactics",
            "The politics of the school pickup line",
            "Organic everything anxiety"
        ]
    ),
    
    "foodie_comedian": PersonaTemplate(
        name="Foodie Comedian",
        description="Food-obsessed with humor around cooking, eating, and food culture",
        humor_style="food-focused, sensory, culturally aware",
        expertise_areas=["cooking", "restaurants", "food trends", "kitchen disasters"],
        demographic_hints={"interests": ["cooking", "dining", "food culture"]},
        prompt_style="Create food-related humor with culinary knowledge",
        examples=[
            "My relationship with sourdough starter",
            "Artisanal air",
            "The emotional journey of a burnt dinner"
        ]
    ),
    
    "college_survivor": PersonaTemplate(
        name="College Survivor",
        description="Post-college humor about student life, debt, and entering the real world",
        humor_style="self-deprecating, relatable, financially anxious",
        expertise_areas=["college life", "student debt", "job hunting", "young adult struggles"],
        demographic_hints={"age_range": "22-28", "education": "college", "financial_status": "struggling"},
        prompt_style="Create relatable humor about post-college life and young adult struggles",
        examples=[
            "My degree in underwater basket weaving",
            "Living on ramen and false hope",
            "The job market's cruel sense of humor"
        ]
    ),
    
    "absurdist_artist": PersonaTemplate(
        name="Absurdist Artist",
        description="Creative, weird, artistically-inclined with surreal humor",
        humor_style="surreal, artistic, avant-garde",
        expertise_areas=["art", "creativity", "surrealism", "avant-garde"],
        demographic_hints={"occupation": "creative", "artistic_inclination": True},
        prompt_style="Create surreal, artistic humor that's weird but brilliant",
        examples=[
            "The color blue's midlife crisis",
            "My art degree's existential awakening",
            "Performance art that only pigeons understand"
        ]
    ),
    
    "wordplay_master": PersonaTemplate(
        name="Wordplay Master",
        description="Expert in puns, word games, and linguistic humor",
        humor_style="punny, clever, linguistically sophisticated",
        expertise_areas=["puns", "wordplay", "linguistics", "clever language"],
        demographic_hints={"language_skills": "high", "pun_tolerance": "maximum"},
        prompt_style="Create clever wordplay and puns that are actually funny",
        examples=[
            "My punintentional humor",
            "A spelling bee's worst nightmare",
            "Autocorrect's passive-aggressive suggestions"
        ]
    )
}

# AI Comedian Personas (Dynamic/Generated)
AI_COMEDIAN_PERSONAS = {
    "adaptive_humor_expert": PersonaTemplate(
        name="Adaptive Humor Expert",
        description="AI comedian that learns and adapts to user preferences in real-time",
        humor_style="adaptive, learning, personalized",
        expertise_areas=["personalization", "adaptation", "user preferences", "AI humor"],
        demographic_hints={"ai_generated": True, "adaptive": True},
        prompt_style="Generate humor that adapts to user preferences and learns from feedback",
        examples=[
            "Something perfectly tailored to you",
            "Humor that gets better with time",
            "AI that actually understands your jokes"
        ]
    )
}

def get_all_personas() -> Dict[str, PersonaTemplate]:
    """Get all enhanced personas including AI comedians"""
    return {**ENHANCED_HUMOR_PERSONAS, **AI_COMEDIAN_PERSONAS}

def get_ai_comedians() -> Dict[str, PersonaTemplate]:
    """Get only AI comedian personas"""
    return AI_COMEDIAN_PERSONAS

def recommend_personas_for_context(context: str, audience: str, topic: str) -> List[str]:
    """Enhanced persona recommendation with better logic"""
    recommendations = []
    
    # Context-based recommendations
    if "work" in context.lower() or "office" in context.lower():
        recommendations.extend(["office_worker", "suburban_parent"])
    elif "family" in context.lower() or "parent" in context.lower():
        recommendations.extend(["suburban_parent", "dad_humor_enthusiast"])
    elif "game" in context.lower() or "gaming" in context.lower():
        recommendations.extend(["gaming_guru"])
    elif "food" in context.lower() or "cooking" in context.lower():
        recommendations.extend(["foodie_comedian"])
    elif "school" in context.lower() or "college" in context.lower():
        recommendations.extend(["college_survivor"])
    
    # Audience-based recommendations  
    if audience == "friends":
        recommendations.extend(["gaming_guru"])  # FIXED: removed duplicate millennial_memer
    elif audience == "family":
        recommendations.extend(["dad_humor_enthusiast", "suburban_parent"])
    elif audience == "colleagues":
        recommendations.extend(["office_worker"])
    elif audience == "general":
        recommendations.extend(["gen_z_chaos", "gaming_guru"])  # FIXED: removed duplicate millennial_memer
    
    # Topic-based recommendations
    if topic == "technology":
        recommendations.extend(["gaming_guru"])  # FIXED: removed duplicate millennial_memer
    elif topic == "family":
        recommendations.extend(["suburban_parent"])  # FIXED: removed duplicate millennial_memer
    elif topic == "work":
        recommendations.extend(["office_worker"])  # FIXED: removed duplicate millennial_memer
    else:
        recommendations.extend(["dad_humor_enthusiast", "office_worker"])  # FIXED: removed duplicate millennial_memer
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for persona in recommendations:
        if persona not in seen:
            seen.add(persona)
            unique_recommendations.append(persona)
    
    return unique_recommendations[:3]  # Return top 3

def get_personas_by_style(humor_style: str) -> List[str]:
    """Get personas that match a specific humor style"""
    if humor_style == "family_friendly":
        return ["dad_humor_enthusiast", "suburban_parent", "foodie_comedian"]
    elif humor_style == "edgy":
        return ["dark_humor_specialist", "gen_z_chaos", "absurdist_artist"]
    elif humor_style == "geeky":
        return ["gaming_guru", "wordplay_master", "college_survivor"]  # FIXED: removed duplicate millennial_memer
    elif humor_style == "professional":
        return ["office_worker", "suburban_parent"]  # FIXED: removed duplicate millennial_memer
    else:
        return ["dad_humor_enthusiast", "office_worker"]  # FIXED: removed duplicate millennial_memer 