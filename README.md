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

## Evaluation Metrics

### Statistical Humor Evaluator
Implementation: `evaluation/statistical_humor_evaluator.py`

## Experimental Results

### Comparison with Baseline Methods

#### Multi-Agent vs Simple Prompt Engineering

**CrewAI Multi-Agent System:**
```bash
python agent_system/complete_system_demo.py > system_demo_output_clean.txt
```

**Simple Prompt Engineering Baseline:**
```bash
python agent_system/fixed_complete_demo.py > fixed_demo_output.txt
```

**Results & Analysis:**
- Summary: `agent_system/final_results/analysis_summary.txt`
- Detailed Results: `agent_system/final_results/complete_cah_analysis.json`

### Advanced Generation Methods

#### PPLM and RAG Evaluation
**Implementation:** `agent_system/run_full_comprehensive_evaluation.py`

Evaluation of advanced text generation techniques:
- **PPLM (Plug and Play Language Models)**: Real iterative gradient ascent during generation
- **RAG (Retrieval-Augmented Generation)**: Vector database with 50+ fact corpus using ChromaDB
- **Statistical Evaluation**: Literature-based computational metrics

**Results:** `agent_system/comprehensive_evaluation_20250908_144358.csv`

### Personalization Analysis

#### Static vs Dynamic Persona Comparison
**Implementation:** `evaluation/personalization_comparison_metrics.py`

**Results:** `evaluation/static_vs_dynamic/`

- Ground Truth: `evaluation/outputs/` (y_true data)
- Generated Results: `evaluation/outputs/complete_sentences_evaluation_results_20250825_005120.json`
- Generation Code: `agent_system/generate_example_cards_ai.py`



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
