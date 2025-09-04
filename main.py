#!/usr/bin/env python3
"""
Main CAH System Entry Point
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
from datetime import datetime

# from detoxify import Detoxify  # causes memory issues on Render in prod

def simple_content_filter(text: str) -> Dict[str, float]:
    """Simple regex-based content filter (replaces detoxify)"""
    # Basic toxicity patterns
    patterns = {
        'toxicity': [
            r"\b(hate|despise|loathe)\s+(all|every)\s+\w+",
            r"\b(kill|murder|die)\s+(all|every)\s+\w+",
            r"\b\w+\s+(are|is)\s+(evil|scum|trash|garbage)"
        ],
        'profanity': [
            r"\bf[*]?u[*]?c[*]?k\w*",
            r"\bs[*]?h[*]?i[*]?t\w*", 
            r"\bd[*]?a[*]?m[*]?n\w*",
            r"\bb[*]?i[*]?t[*]?c[*]?h\w*"
        ]
    }
    
    scores = {}
    for category, pattern_list in patterns.items():
        score = 0.0
        for pattern in pattern_list:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                score = min(1.0, matches * 0.3)
        scores[category] = score
    
    # Add other categories with 0 scores
    scores.update({
        'severe_toxicity': 0.0,
        'obscene': 0.0,
        'threat': 0.0,
        'insult': 0.0,
        'identity_attack': 0.0
    })
    
    return scores

def check_content_safety(text: str) -> Dict[str, float]:
    """Check if content is safe using simple regex patterns (replaces detoxify)"""
    try:
        results = simple_content_filter(text)
        return results
    except Exception as e:
        print(f"Content filter error: {e}")
        # Fallback: return safe scores
        return {
            'toxicity': 0.0,
            'severe_toxicity': 0.0,
            'obscene': 0.0,
            'threat': 0.0,
            'insult': 0.0,
            'identity_attack': 0.0
        }

# Import AI generation modules
from generation.card_generator import (
    generate_personalized_black_cards,
    generate_personalized_white_cards
)

# Load environment variables
load_dotenv()

app = FastAPI(title="Personalized Cards Against Humanity API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Tag(BaseModel):
    name: str
    weight: float = 1.0

class UserPreferences(BaseModel):
    themes: List[str] = []
    humorStyle: List[str] = []
    customTopics: List[str] = []
    contentFilter: str = "moderate"  # "family-friendly", "moderate", "no-filter"

class BlackCard(BaseModel):
    id: str
    text: str
    pick: int
    tags: Optional[List[str]] = None

class WhiteCard(BaseModel):
    id: str
    text: str
    tags: Optional[List[str]] = None

class GenerateCardsRequest(BaseModel):
    preferences: UserPreferences
    blackCardCount: int = 10
    whiteCardCount: int = 20

class GenerateCardsResponse(BaseModel):
    blackCards: List[BlackCard]
    whiteCards: List[WhiteCard]

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Personalized Cards Against Humanity API"}

@app.post("/generate", response_model=GenerateCardsResponse)
async def generate_cards(request: GenerateCardsRequest):
    """Generate personalized CAH cards based on user preferences"""
    try:
        # Generate black cards
        black_cards = await generate_personalized_black_cards(
            request.preferences, 
            request.blackCardCount
        )
        
        # Generate white cards
        white_cards = await generate_personalized_white_cards(
            request.preferences, 
            request.whiteCardCount
        )
        
        # Apply content filtering if needed
        if request.preferences.contentFilter != "no-filter":
            # For family-friendly mode, use stricter threshold
            threshold = 0.3 if request.preferences.contentFilter == "family-friendly" else 0.7
            
            # Filter black cards
            filtered_black_cards = []
            for card in black_cards:
                results = check_content_safety(card.text)
                # Check if any toxicity score is above threshold
                if not any(score > threshold for score in results.values()):
                    filtered_black_cards.append(card)
            
            # Filter white cards
            filtered_white_cards = []
            for card in white_cards:
                results = check_content_safety(card.text)
                if not any(score > threshold for score in results.values()):
                    filtered_white_cards.append(card)
            
            # If we filtered too many, generate more
            if len(filtered_black_cards) < request.blackCardCount / 2:
                # In a real implementation, we would generate more cards
                # For now, just use the less toxic ones
                filtered_black_cards = sorted(
                    black_cards, 
                    key=lambda x: max(check_content_safety(x.text).values())
                )[:request.blackCardCount]
            
            if len(filtered_white_cards) < request.whiteCardCount / 2:
                filtered_white_cards = sorted(
                    white_cards, 
                    key=lambda x: max(check_content_safety(x.text).values())
                )[:request.whiteCardCount]
            
            black_cards = filtered_black_cards
            white_cards = filtered_white_cards
        
        return GenerateCardsResponse(
            blackCards=black_cards[:request.blackCardCount],
            whiteCards=white_cards[:request.whiteCardCount]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 