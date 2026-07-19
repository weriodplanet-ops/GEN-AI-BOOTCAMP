"""
Unit tests for ProductAnalyzer
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import os

try:
    from src.product_analyzer import ProductAnalyzer
except ImportError:
    from product_analyzer import ProductAnalyzer


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Create a simple colored image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)


class TestProductAnalyzer:
    """Test suite for ProductAnalyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = ProductAnalyzer(model_type="blip")
        assert analyzer is not None
        assert analyzer.model_type == "blip"
    
    def test_config_loading(self):
        """Test configuration loading"""
        analyzer = ProductAnalyzer(model_type="blip")
        assert "model" in analyzer.config
        assert "categories" in analyzer.config
        assert "description" in analyzer.config
    
    def test_image_loading(self, sample_image):
        """Test image loading functionality"""
        analyzer = ProductAnalyzer(model_type="blip")
        image = analyzer._load_image(sample_image)
        assert image is not None
        assert image.mode == "RGB"
    
    def test_invalid_image_path(self):
        """Test error handling for invalid image path"""
        analyzer = ProductAnalyzer(model_type="blip")
        with pytest.raises(FileNotFoundError):
            analyzer._load_image("/nonexistent/path/image.jpg")
    
    def test_category_prediction(self, sample_image):
        """Test category prediction"""
        analyzer = ProductAnalyzer(model_type="blip")
        image = analyzer._load_image(sample_image)
        categories, scores = analyzer._predict_category_blip(image)
        
        assert len(categories) == 3
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
    
    def test_description_generation(self, sample_image):
        """Test description generation"""
        analyzer = ProductAnalyzer(model_type="blip")
        image = analyzer._load_image(sample_image)
        description = analyzer._generate_description(image)
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_full_analysis(self, sample_image):
        """Test complete product analysis"""
        analyzer = ProductAnalyzer(model_type="blip")
        result = analyzer.analyze(sample_image)
        
        assert "category" in result
        assert "description" in result
        assert "category_confidence" in result
        assert "alternatives" in result
        assert result["category_confidence"] >= 0
        assert result["category_confidence"] <= 1
