# Setup Guide

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/weriodplanet-ops/GEN-AI-BOOTCAMP.git
cd GEN-AI-BOOTCAMP
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API Server
```bash
python src/api/app.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Interactive Documentation
Visit: http://localhost:8000/docs

### Analyze Single Image
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@path/to/image.jpg"
```

### Get Categories
```bash
curl http://localhost:8000/categories
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_analyzer.py

# Run with verbose output
pytest tests/ -v
```

## Using in Python

```python
from src.product_analyzer import ProductAnalyzer

# Initialize
analyzer = ProductAnalyzer(model_type="blip")

# Analyze image
result = analyzer.analyze("path/to/product.jpg")
print(result['category'])
print(result['description'])
```

## Configuration

Edit `config.yaml` to customize:
- Model selection (CLIP, BLIP, Ensemble)
- Target image size
- Category list
- Description parameters
- Performance targets

## Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Then open `01_exploration.ipynb` for interactive exploration.

## GPU Support

The system automatically uses GPU if available. To force CPU:

```python
config["model"]["device"] = "cpu"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config.yaml
- Use CPU mode
- Reduce image resolution

### Model Download Issues
- Ensure internet connection
- Hugging Face models will download to `~/.cache/huggingface/`

### Import Errors
- Verify virtual environment is activated
- Check all dependencies installed: `pip install -r requirements.txt`
