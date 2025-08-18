#!/usr/bin/env python3
"""
Retrieval-Augmented Humor Generation
Implements retrieval component as discussed in literature (Horvitz et al.)
"""

import json
import random
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class RetrievalContext:
    """Context retrieved for humor generation"""
    entities: List[str]
    facts: List[str]
    related_jokes: List[str]
    topical_info: List[str]
    confidence_score: float

@dataclass
class HumorTemplate:
    """Template-based humor structure"""
    pattern: str
    example: str
    category: str
    success_rate: float

class HumorKnowledgeBase:
    """Knowledge base for retrieval-augmented humor"""
    
    def __init__(self):
        # Mock knowledge base - in practice would be populated from datasets
        self.entities_facts = {
            "taxes": [
                "Annual obligation that makes adults cry",
                "The reason accountants exist",
                "What turns math into emotional trauma"
            ],
            "vegetables": [
                "Expensive green things parents force on children",
                "Proof that healthy choices cost more",
                "Nature's way of making salad sad"
            ],
            "adulthood": [
                "When you realize your parents were just winging it",
                "The realization that no one actually knows what they're doing",
                "When excitement means new kitchen appliances"
            ],
            "work": [
                "Place where productivity goes to die in meetings",
                "Where coffee becomes a food group",
                "Eight hours of pretending to be busy"
            ],
            "family": [
                "People who know all your embarrassing stories",
                "DNA-bound reality TV show cast",
                "Proof that you can't choose your relatives"
            ]
        }
        
        self.joke_templates = [
            HumorTemplate(
                pattern="The worst part about {topic} is {unexpected_twist}",
                example="The worst part about adulthood is realizing vegetables are expensive",
                category="complaint_humor",
                success_rate=0.72
            ),
            HumorTemplate(
                pattern="{topic} is like {unexpected_comparison} because {reason}",
                example="Taxes are like exercise because everyone knows they should do it but procrastinates",
                category="comparison_humor", 
                success_rate=0.68
            ),
            HumorTemplate(
                pattern="I used to think {topic} was {expectation}, but now I know it's {reality}",
                example="I used to think adults had everything figured out, but now I know we're all improvising",
                category="realization_humor",
                success_rate=0.75
            )
        ]
        
        self.contextual_jokes = {
            "work": [
                "My productivity is like a graph: mostly flat with occasional spikes of panic",
                "I'm not procrastinating, I'm doing extensive quality control on my procrastination",
                "Work meetings: where minutes are kept but hours are lost"
            ],
            "family": [
                "Family dinners: where your childhood failures get annual reviews",
                "Relatives: proof that genetics has a sense of humor",
                "Family photos: documenting decades of awkward smiles"
            ],
            "adulthood": [
                "Adulthood is just childhood but with bills and back pain",
                "Growing up means your bed time is self-imposed disappointment",
                "Adult life: where weekends feel like commercial breaks"
            ]
        }

class RetrievalAugmentedHumorGenerator:
    """
    Retrieval-augmented humor generation following literature recommendations
    Literature: Horvitz et al. - "retrieval can support more dynamic personalization"
    """
    
    def __init__(self):
        self.knowledge_base = HumorKnowledgeBase()
        self.retrieval_cache = {}
        
    def extract_entities(self, context: str) -> List[str]:
        """Extract key entities from context for retrieval"""
        # Simple entity extraction - in practice would use NER
        entities = []
        
        # Common humor topics
        humor_entities = [
            "work", "job", "boss", "office", "colleague",
            "family", "parent", "child", "sibling", "relative",
            "tax", "money", "bill", "expense", "budget",
            "adult", "adulthood", "grown-up", "responsibility",
            "food", "vegetable", "cooking", "diet", "health"
        ]
        
        context_lower = context.lower()
        for entity in humor_entities:
            if entity in context_lower:
                entities.append(entity)
                
        return entities
    
    async def retrieve_context(self, prompt: str, entities: List[str]) -> RetrievalContext:
        """
        Retrieve relevant context for humor generation
        Literature: "retrieval of topical facts or existing jokes, then rewrites"
        """
        # Extract and retrieve facts
        facts = []
        related_jokes = []
        topical_info = []
        
        for entity in entities:
            # Get facts about entity
            if entity in self.knowledge_base.entities_facts:
                facts.extend(self.knowledge_base.entities_facts[entity][:2])
            
            # Get related jokes
            if entity in self.knowledge_base.contextual_jokes:
                related_jokes.extend(self.knowledge_base.contextual_jokes[entity][:2])
            
            # Add topical information
            topical_info.append(f"Current context involves {entity}")
        
        # Calculate confidence based on retrieval success
        confidence = min(1.0, (len(facts) + len(related_jokes)) / 5.0)
        
        return RetrievalContext(
            entities=entities,
            facts=facts,
            related_jokes=related_jokes,
            topical_info=topical_info,
            confidence_score=confidence
        )
    
    def select_best_template(self, context: RetrievalContext) -> Optional[HumorTemplate]:
        """Select most appropriate humor template based on context"""
        # Score templates based on context match
        template_scores = []
        
        for template in self.knowledge_base.joke_templates:
            score = template.success_rate
            
            # Boost score if template category matches context
            if any(entity in template.example.lower() for entity in context.entities):
                score += 0.1
                
            template_scores.append((template, score))
        
        # Select highest scoring template
        if template_scores:
            return max(template_scores, key=lambda x: x[1])[0]
        
        return None
    
    async def generate_with_retrieval(
        self, 
        prompt: str, 
        audience: str = "general",
        style: str = "witty"
    ) -> Dict[str, Any]:
        """
        Generate humor using retrieval augmentation
        
        Process:
        1. Extract entities from prompt
        2. Retrieve relevant context and jokes
        3. Select appropriate template
        4. Generate humor using retrieved information
        """
        
        # Step 1: Entity extraction
        entities = self.extract_entities(prompt)
        
        # Step 2: Retrieval
        context = await self.retrieve_context(prompt, entities)
        
        # Step 3: Template selection
        template = self.select_best_template(context)
        
        # Step 4: Generation
        generated_humor = await self._generate_from_template_and_context(
            prompt, template, context, audience, style
        )
        
        return {
            "generated_text": generated_humor,
            "retrieval_context": context,
            "template_used": template,
            "generation_method": "retrieval_augmented",
            "entities_found": entities,
            "confidence": context.confidence_score
        }
    
    async def _generate_from_template_and_context(
        self,
        prompt: str,
        template: Optional[HumorTemplate],
        context: RetrievalContext,
        audience: str,
        style: str
    ) -> str:
        """Generate humor using template and retrieved context"""
        
        if not template or not context.facts:
            # Fallback to simple generation
            return await self._generate_fallback(prompt, context)
        
        # Use template with retrieved facts
        primary_entity = context.entities[0] if context.entities else "life"
        
        if template.pattern == "The worst part about {topic} is {unexpected_twist}":
            if context.facts:
                twist = context.facts[0].lower()
                return f"The worst part about {primary_entity} is {twist}"
        
        elif template.pattern == "{topic} is like {unexpected_comparison} because {reason}":
            if len(context.facts) >= 2:
                comparison = "a complicated math problem"
                reason = context.facts[0].lower()
                return f"{primary_entity.title()} is like {comparison} because {reason}"
        
        elif template.pattern == "I used to think {topic} was {expectation}, but now I know it's {reality}":
            if context.facts:
                expectation = "simple"
                reality = context.facts[0].lower()
                return f"I used to think {primary_entity} was {expectation}, but now I know it's {reality}"
        
        # If template matching fails, use related jokes
        if context.related_jokes:
            return context.related_jokes[0]
        
        return await self._generate_fallback(prompt, context)
    
    async def _generate_fallback(self, prompt: str, context: RetrievalContext) -> str:
        """Fallback generation when template/retrieval fails"""
        fallbacks = [
            "Something unexpectedly relatable about this situation",
            "The uncomfortable truth everyone thinks but doesn't say",
            "A surprisingly accurate observation wrapped in humor",
            "The kind of thing that makes you laugh then question your life choices"
        ]
        
        if context.related_jokes:
            return random.choice(context.related_jokes)
        
        return random.choice(fallbacks)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance"""
        total_entities = len(self.knowledge_base.entities_facts)
        total_templates = len(self.knowledge_base.joke_templates)
        avg_template_success = sum(t.success_rate for t in self.knowledge_base.joke_templates) / total_templates
        
        return {
            "knowledge_base_size": {
                "entities": total_entities,
                "templates": total_templates,
                "jokes_per_entity": sum(len(jokes) for jokes in self.knowledge_base.contextual_jokes.values()) / len(self.knowledge_base.contextual_jokes)
            },
            "average_template_success_rate": avg_template_success,
            "retrieval_coverage": list(self.knowledge_base.entities_facts.keys()),
            "template_categories": list(set(t.category for t in self.knowledge_base.joke_templates))
        }

# Global instance
retrieval_humor_generator = RetrievalAugmentedHumorGenerator() 