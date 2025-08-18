from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import random
from dotenv import load_dotenv
from detoxify import Detoxify

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

# Content safety check dependency
def check_content_safety(text: str, threshold: float = 0.5) -> Dict[str, float]:
    """Check if content is safe using Detoxify"""
    results = Detoxify('original').predict(text)
    return results

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
                results = check_content_safety(card.text, threshold)
                # Check if any toxicity score is above threshold
                if not any(score > threshold for score in results.values()):
                    filtered_black_cards.append(card)
            
            # Filter white cards
            filtered_white_cards = []
            for card in white_cards:
                results = check_content_safety(card.text, threshold)
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