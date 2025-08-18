# Personalized Cards Against Humanity

A web application that generates personalized Cards Against Humanity cards using NLP techniques.

## Project Structure

- **python-backend/**: Backend services for model training, evaluation, and inference


## Technology Stack

### Frontend
- **Next.js**: React framework for server-side rendering and static site generation
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React**: UI component library

### Backend
- **Python**: For AI/ML components
- **FastAPI**: API framework for Python
- **Hugging Face Transformers**: For fine-tuning and deploying language models
- **PyTorch**: Deep learning framework

## Approaches

- **Fine-tuned Models**: T5, BART fine-tuned on CAH card datasets
- **Semantic Evaluation**: Using  DistilBERT for humor scoring
- **Content Filtering**: Ensuring generated humor is appropriate using Detofixy

## Model Evaluation

The project includes comprehensive evaluation of different models:
- ROUGE metrics for text similarity
- Humor scoring based on trained classifiers (https://huggingface.co/mohameddhiab/humor-no-humor)
- Generation time and efficiency metrics

## Installation and Setup

### Frontend
```bash
cd cah-app
npm install
npm run dev
```

### Backend
```bash
cd python-backend
pip install -r requirements.txt
python src/finetune_bart_model.py
```

## Acknowledgments
