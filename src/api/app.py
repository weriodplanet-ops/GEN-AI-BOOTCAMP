"""
FastAPI application for serving the ProductAnalyzer
Provides REST API endpoints for product image analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from io import BytesIO
from PIL import Image
import tempfile
import os

from product_analyzer import ProductAnalyzer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Product Analyzer API",
    description="AI system for automatic product categorization and description generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (loaded once at startup)
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global analyzer
    logger.info("Loading ProductAnalyzer models...")
    try:
        analyzer = ProductAnalyzer(model_type="blip")
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Multimodal Product Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": analyzer is not None
    }


@app.post("/analyze")
async def analyze_product(file: UploadFile = File(...)):
    """
    Analyze a product image and generate category and description
    
    Args:
        file: Product image file (JPG, PNG, WebP)
    
    Returns:
        JSON with:
        - category: Predicted product category
        - category_confidence: Confidence score (0-1)
        - description: Generated product description
        - alternatives: Top 3 category alternatives
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Save to temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            image.save(tmp_path, "JPEG")
        
        # Analyze
        result = analyzer.analyze(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple product images
    
    Args:
        files: List of product image files
    
    Returns:
        List of analysis results
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    temp_paths = []
    
    try:
        # Save all images to temporary files
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
                results.append({
                    "file": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            try:
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    image.save(tmp_path, "JPEG")
                    temp_paths.append(tmp_path)
                    
                    # Analyze
                    result = analyzer.analyze(tmp_path)
                    result["file"] = file.filename
                    results.append(result)
            
            except Exception as e:
                results.append({
                    "file": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content=results)
    
    finally:
        # Cleanup temporary files
        for tmp_path in temp_paths:
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.get("/categories")
async def get_categories():
    """Get list of supported product categories"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "categories": analyzer.config["categories"]["predefined"]
    }


@app.get("/config")
async def get_config():
    """Get model and API configuration"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model": analyzer.config["model"],
        "description": analyzer.config["description"],
        "categories": {
            "count": len(analyzer.config["categories"]["predefined"]),
            "list": analyzer.config["categories"]["predefined"]
        },
        "business_metrics": analyzer.config.get("business", {})
    }


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics and benchmarks"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "target_metrics": analyzer.config.get("business", {}),
        "benchmarks": analyzer.config.get("benchmarks", {}),
        "model_type": analyzer.model_type
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
