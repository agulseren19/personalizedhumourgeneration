#!/usr/bin/env python3
"""
Simple Prompt Generator - Basic CAH Response Generation
Just a basic OpenAI API call without any advanced features
"""

import openai
import time
import os
from dotenv import load_dotenv

load_dotenv()

def simple_generate(context, audience="general"):
    """Simple prompt generation - just one API call"""
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""Complete this Cards Against Humanity card:

Black Card: "{context}"
Audience: {audience}

Respond with a witty white card response:"""
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=50
    )
    
    generation_time = time.time() - start_time
    result = response.choices[0].message.content.strip()
    
    return {
        'response': result,
        'time': generation_time,
        'model': 'gpt-3.5-turbo',
        'tokens': response.usage.total_tokens
    }

def run_comparison():
    """Run comparison test with multiple contexts"""
    
    test_contexts = [
        {"context": "What's the best legal excuse for being late? _____", "audience": "colleagues"},
        {"context": "What did I pack for my kid's lunch? _____", "audience": "family"},
        {"context": "What's my secret gaming strategy? _____", "audience": "friends"},
        {"context": "What's in my browser history? _____", "audience": "general"},
        {"context": "What's my biggest work mistake? _____", "audience": "colleagues"}
    ]
    
    print("SIMPLE PROMPT GENERATOR COMPARISON")
    print("="*50)
    print("Basic single-response generation (no agents, no learning, no evaluation)")
    print()
    
    total_time = 0
    results = []
    
    for i, test in enumerate(test_contexts, 1):
        print(f"Test {i}: {test['context']}")
        print(f"Audience: {test['audience']}")
        
        try:
            result = simple_generate(test['context'], test['audience'])
            total_time += result['time']
            results.append(result)
            
            print(f"Response: \"{result['response']}\"")
            print(f"Time: {result['time']:.2f}s")
            print(f"Model: {result['model']}")
            print(f"Tokens: {result['tokens']}")
            print()
            
        except Exception as e:
            print(f"ERROR: {e}")
            print()
    
    print("="*50)
    print("SIMPLE GENERATOR SUMMARY:")
    print(f"Total Tests: {len(test_contexts)}")
    print(f"Successful: {len(results)}")
    print(f"Average Time: {total_time/len(results):.2f}s per response")
    print(f"Total Time: {total_time:.2f}s")
    print()
    print("LIMITATIONS:")
    print("• No personalization or learning")
    print("• No quality evaluation") 
    print("• No persona-based humor styles")
    print("• No user preference tracking")
    print("• No multi-model intelligence")
    print("• No group context support")
    print("• No fallback systems")
    print("• No cloud storage or analytics")
    print("="*50)

if __name__ == "__main__":
    run_comparison() 