# Personalized Cards Against Humanity

A web application that generates and evaluates personalized Cards Against Humanity cards using NLP techniques and multi-agent systems.

## Deployment

The application is currently deployed and accessible at:
https://personalizedhumourgenerationcah.vercel.app/cah

- **Backend**: Deployed to Render
- **Frontend**: Deployed to Vercel

## Project Structure

- **agent_system/**: Main backend system with multi-agent architecture, APIs, and game logic
- **nextjs-boilerplate-main/**: Frontend Next.js application
- **evaluation/**: Comprehensive evaluation metrics implementation based on literature review
- **python-backend/**: Fine-tuned BART and T5 model implementations
- **data/**: Dataset used for fine-tuning models

## Technology Stack

### Frontend
- **Next.js**: React framework for server-side rendering and static site generation
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React**: UI component library

### Backend
- **Python**: For AI/ML components
- **FastAPI**: API framework for Python
- **CrewAI**: Multi-agent system framework
- **Hugging Face Transformers**: For fine-tuning and deploying language models
- **PyTorch**: Deep learning framework
- **PostgreSQL**: Database system

## Approaches

- **Multi-Agent System**: CrewAI-based agents for humor generation and evaluation
- **Fine-tuned Models**: T5, BART fine-tuned on CAH card datasets
- **Content Filtering**: Ensuring generated humor is appropriate using Detoxify
- **Personalization**: Dynamic persona generation and user preference learning using user embeddings
- **Evaluation**: Humour, Surprisal, Ambiguity, Distinctiveness, Creativity, Semantic Diversity, Information-Theory Metrics, Persona–Card Similarity (PaCS), F1, MSE


## Local Development

### Running the System Locally

To run the complete system locally:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export USE_CREWAI_AGENTS=true
python start_cah_working.py
```

### Frontend Setup
```bash
cd nextjs-boilerplate-main
npm install
npm run dev
```

### Backend Setup
```bash
cd agent_system
pip install -r requirements.txt
python api/cah_crewai_api.py
```

### Fine-tuned Models
```bash
cd python-backend
pip install -r requirements.txt
python src/finetune_bart_model.py
```

## Acknowledgments
