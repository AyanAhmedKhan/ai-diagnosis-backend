# Backend - GI Endoscopy AI Diagnostic API

FastAPI backend with Grad-CAM explainability for GI endoscopy image analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your TorchScript models in `models/`:
   - `deit3_best_traced.pt`
   - `vit_best_traced.pt`

3. Run the server:
```bash
uvicorn app_gradcam:app --reload --port 8000
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Upload image for diagnosis

## Model Requirements

Models should be TorchScript (.pt) files that:
- Accept: `[1, 3, 384, 384]` tensor
- Output: `[1, 23]` logits

## Docker

```bash
docker build -t gi-endoscopy-backend .
docker run -p 8000:8000 -v $(pwd)/models:/app/models gi-endoscopy-backend
```

