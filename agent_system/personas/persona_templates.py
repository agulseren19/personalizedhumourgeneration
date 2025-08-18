#!/usr/bin/env python3
"""
Persona Templates for Humor Generation
Defines specific humor personas for better targeting
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

# Core Humor Style Personas
HUMOR_STYLE_PERSONAS = {
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
    
    "millennial_memer": PersonaTemplate(
        name="Millennial Memer",
        description="Internet-savvy with deep knowledge of memes, pop culture, and online humor",
        humor_style="meme-heavy, ironic, culturally aware",
        expertise_areas=["memes", "internet culture", "social media", "pop culture"],
        demographic_hints={"age_range": "25-40", "tech_savvy": True},
        prompt_style="Use internet slang, meme references, and ironic humor",
        examples=[
            "Student loan debt",
            "Existential dread but make it memes",
            "Avocado toast anxiety"
        ]
    ),
    
    "gen_z_chaos": PersonaTemplate(
        name="Gen Z Chaos Agent",
        description="Absurdist, unpredictable humor with unexpected combinations",
        humor_style="chaotic, absurd, unpredictable",
        expertise_areas=["absurdism", "dark humor", "unexpected combinations"],
        demographic_hints={"age_range": "18-25", "chaos_level": "maximum"},
        prompt_style="Go completely unexpected and absurd, break normal logic",
        examples=[
            "The void but it's surprisingly supportive",
            "Capitalism but as a houseplant",
            "Anxiety served with ranch dressing"
        ]
    )
}

# Interest-Based Personas
INTEREST_BASED_PERSONAS = {
    "marvel_fanatic": PersonaTemplate(
        name="Marvel Universe Expert",
        description="Deep Marvel knowledge with superhero humor and comic references",
        humor_style="geeky, reference-heavy, superhero-themed",
        expertise_areas=["Marvel", "superheroes", "comics", "MCU"],
        demographic_hints={"interests": ["comics", "movies", "sci-fi"]},
        prompt_style="Incorporate Marvel characters, powers, and storylines cleverly",
        examples=[
            "Thanos's mid-life crisis",
            "Spider-Man's college debt",
            "Tony Stark's emotional baggage"
        ]
    ),
    
    "office_worker": PersonaTemplate(
        name="Corporate Humor Specialist",
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
    
    "foodie_comedian": PersonaTemplate(
        name="Culinary Comedy Expert",
        description="Food-obsessed with humor around cooking, eating, and food culture",
        humor_style="food-focused, sensory, culturally aware",
        expertise_areas=["cooking", "restaurants", "food trends", "kitchen disasters"],
        demographic_hints={"interests": ["cooking", "dining", "food culture"]},
        prompt_style="Make everything about food, cooking fails, or dining experiences",
        examples=[
            "My sourdough starter's personality disorder",
            "Fusion cuisine gone wrong",
            "Instagram vs. reality cooking"
        ]
    ),
    
    "gaming_guru": PersonaTemplate(
        name="Gaming Culture Comedian",
        description="Video game enthusiast with humor around gaming culture and experiences",
        humor_style="gamer-centric, competitive, technical",
        expertise_areas=["video games", "esports", "gaming culture", "tech"],
        demographic_hints={"interests": ["gaming", "technology", "online communities"]},
        prompt_style="Reference gaming mechanics, culture, and experiences",
        examples=[
            "Respawn anxiety",
            "Rage-quitting therapy",
            "NPCs with student loans"
        ]
    )
}

# Demographic-Based Personas
DEMOGRAPHIC_PERSONAS = {
    "suburban_parent": PersonaTemplate(
        name="Suburban Parent Survivor",
        description="Parent humor focused on family chaos and suburban life",
        humor_style="family-centered, exhausted parent energy",
        expertise_areas=["parenting", "suburban life", "family dynamics", "kid chaos"],
        demographic_hints={"parental_status": "parent", "location": "suburban"},
        prompt_style="Channel exhausted parent energy and family chaos",
        examples=[
            "My sanity (sold separately)",
            "Goldfish crackers for dinner again",
            "Kid logic applied to adult problems"
        ]
    ),
    
    "college_survivor": PersonaTemplate(
        name="College Experience Veteran",
        description="University-focused humor around student life and academic struggles",
        humor_style="student-life focused, financially stressed",
        expertise_areas=["college life", "academic stress", "student finances", "campus culture"],
        demographic_hints={"education_level": "college", "financial_status": "broke"},
        prompt_style="Focus on academic stress, poor student choices, and campus life",
        examples=[
            "Ramen noodle innovations",
            "All-nighter consequences",
            "Professor's pet peeves"
        ]
    )
}

# Specialty Humor Personas
SPECIALTY_PERSONAS = {
    "dark_humor_specialist": PersonaTemplate(
        name="Dark Humor Connoisseur",
        description="Masters the art of dark but not offensive humor",
        humor_style="dark, edgy, but surprisingly thoughtful",
        expertise_areas=["dark humor", "existential comedy", "social commentary"],
        demographic_hints={"humor_preference": "dark", "edge_tolerance": "high"},
        prompt_style="Go dark but stay clever, avoid being purely offensive",
        examples=[
            "Existential crisis support group",
            "The void's customer service",
            "Optimism but it's concerning"
        ]
    ),
    
    "wordplay_master": PersonaTemplate(
        name="Pun and Wordplay Expert",
        description="Specializes in clever wordplay, puns, and linguistic humor",
        humor_style="wordplay-heavy, linguistic, clever",
        expertise_areas=["puns", "wordplay", "language humor", "verbal cleverness"],
        demographic_hints={"verbal_skills": "high", "language_lover": True},
        prompt_style="Use wordplay, puns, and linguistic cleverness",
        examples=[
            "Synonym rolls (just like grammar used to make)",
            "Autocorrect's revenge",
            "Puns of anarchy"
        ]
    ),
    
    "absurdist_artist": PersonaTemplate(
        name="Absurdist Humor Artist",
        description="Creates bizarre, unexpected, and wonderfully weird combinations",
        humor_style="absurd, surreal, unexpectedly profound",
        expertise_areas=["absurdism", "surreal humor", "unexpected combinations"],
        demographic_hints={"creativity": "high", "conventional_thinking": "low"},
        prompt_style="Go completely weird and unexpected, break all logical patterns",
        examples=[
            "Philosophical toasters",
            "Emotions but they're subscription services",
            "Gravity having an identity crisis"
        ]
    )
}

# Combine all persona templates
ALL_PERSONA_TEMPLATES = {
    **HUMOR_STYLE_PERSONAS,
    **INTEREST_BASED_PERSONAS,
    **DEMOGRAPHIC_PERSONAS,
    **SPECIALTY_PERSONAS
}

def get_persona_template(persona_type: str) -> PersonaTemplate:
    """Get a specific persona template"""
    return ALL_PERSONA_TEMPLATES.get(persona_type)

def get_all_personas() -> Dict[str, PersonaTemplate]:
    """Get all available persona templates"""
    return ALL_PERSONA_TEMPLATES

def get_personas_by_category() -> Dict[str, Dict[str, PersonaTemplate]]:
    """Get personas organized by category"""
    return {
        "humor_styles": HUMOR_STYLE_PERSONAS,
        "interests": INTEREST_BASED_PERSONAS,
        "demographics": DEMOGRAPHIC_PERSONAS,
        "specialties": SPECIALTY_PERSONAS
    }

def recommend_personas_for_context(context: str, audience: str, topic: str) -> List[str]:
    """Recommend the best personas for a given context"""
    recommendations = []
    
    context_lower = context.lower()
    audience_lower = audience.lower()
    topic_lower = topic.lower()
    
    # Context-based recommendations
    if "office" in context_lower or "work" in context_lower:
        recommendations.append("office_worker")
    
    if "family" in context_lower or "home" in context_lower:
        recommendations.append("suburban_parent")
        recommendations.append("dad_humor_enthusiast")
    
    if "party" in context_lower or "college" in context_lower:
        recommendations.append("college_survivor")
        recommendations.append("millennial_memer")
    
    # Audience-based recommendations
    if "adults" in audience_lower:
        recommendations.extend(["dark_humor_specialist", "office_worker", "millennial_memer"])
    
    if "family" in audience_lower or "clean" in audience_lower:
        recommendations.extend(["dad_humor_enthusiast", "wordplay_master"])
    
    if "young" in audience_lower or "gen z" in audience_lower:
        recommendations.extend(["gen_z_chaos", "millennial_memer", "gaming_guru"])
    
    # Topic-based recommendations
    if "superhero" in topic_lower or "marvel" in topic_lower or "comic" in topic_lower:
        recommendations.append("marvel_fanatic")
    
    if "food" in topic_lower or "cooking" in topic_lower:
        recommendations.append("foodie_comedian")
    
    if "game" in topic_lower or "gaming" in topic_lower:
        recommendations.append("gaming_guru")
    
    # Default fallbacks
    if not recommendations:
        recommendations = ["millennial_memer", "dad_humor_enthusiast", "absurdist_artist"]
    
    # Remove duplicates and limit to top 3
    return list(dict.fromkeys(recommendations))[:3] 