"""
Main Product Analyzer Module
Combines CLIP and BLIP models for multimodal product understanding
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductAnalyzer:
    """
    Multimodal AI system for analyzing product images and generating:
    - Product categories
    - Product descriptions
    """
    
    def __init__(self, config_path: str = "config.yaml", model_type: str = "blip"):
        """
        Initialize the ProductAnalyzer
        
        Args:
            config_path: Path to configuration YAML file
            model_type: Type of model to use ("clip", "blip", or "ensemble")
        """
        self.config = self._load_config(config_path)
        self.model_type = model_type
        self.device = torch.device(self.config["model"]["device"] if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.clip_processor = None
        self.clip_model = None
        self.blip_processor = None
        self.blip_model = None
        
        self._initialize_models()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_models(self):
        """Initialize CLIP and/or BLIP models based on configuration"""
        logger.info(f"Initializing {self.model_type} model(s)...")
        
        if self.model_type in ["clip", "ensemble"]:
            self._initialize_clip()
        
        if self.model_type in ["blip", "ensemble"]:
            self._initialize_blip()
    
    def _initialize_clip(self):
        """Initialize CLIP model for zero-shot classification"""
        try:
            model_name = f"openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
            logger.info("✓ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _initialize_blip(self):
        """Initialize BLIP model for image captioning and description generation"""
        try:
            model_name = "Salesforce/blip-image-captioning-base"
            self.blip_processor = BlipProcessor.from_pretrained(model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            self.blip_model.eval()
            logger.info("✓ BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Image loaded: {image_path} (Size: {image.size})")
        return image
    
    def _predict_category_blip(self, image: Image.Image) -> Tuple[List[str], List[float]]:
        """
        Predict product category using BLIP + text matching
        
        Returns:
            - List of top 3 category predictions
            - List of corresponding confidence scores
        """
        categories = self.config["categories"]["predefined"]
        
        with torch.no_grad():
            # Generate image caption first
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            caption_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(caption_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
        
        # Simple heuristic-based categorization (can be improved with embeddings)
        scores = self._compute_category_scores(caption, categories)
        top_indices = np.argsort(scores)[-3:][::-1]
        
        top_categories = [categories[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        return top_categories, top_scores
    
    def _predict_category_clip(self, image: Image.Image) -> Tuple[List[str], List[float]]:
        """
        Predict product category using CLIP zero-shot classification
        
        Returns:
            - List of top 3 category predictions
            - List of corresponding confidence scores
        """
        categories = self.config["categories"]["predefined"]
        
        # Create text prompts for each category
        text_prompts = [f"a product in the {cat} category" for cat in categories]
        
        with torch.no_grad():
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**image_inputs)
            
            text_inputs = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Compute similarity scores
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features @ text_features.T) * 100
            scores = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        top_indices = np.argsort(scores)[-3:][::-1]
        top_categories = [categories[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        return top_categories, top_scores
    
    def _generate_description(self, image: Image.Image) -> str:
        """
        Generate product description using BLIP
        
        Returns:
            Product description string
        """
        with torch.no_grad():
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate description with configured parameters
            description_ids = self.blip_model.generate(
                **inputs,
                max_length=self.config["description"]["max_length"],
                min_length=self.config["description"]["min_length"],
                num_beams=self.config["description"]["num_beams"],
                temperature=self.config["description"]["temperature"],
            )
            
            description = self.blip_processor.decode(description_ids[0], skip_special_tokens=True)
        
        return description
    
    def _compute_category_scores(self, caption: str, categories: List[str]) -> np.ndarray:
        """
        Compute category scores based on caption and categories
        This is a simple heuristic - can be improved with embedding similarity
        """
        scores = np.zeros(len(categories))
        caption_lower = caption.lower()
        
        for i, category in enumerate(categories):
            # Simple keyword matching
            if any(keyword in caption_lower for keyword in category.lower().split()):
                scores[i] = 0.5
        
        # If no match found, assign uniform distribution
        if scores.sum() == 0:
            scores = np.ones(len(categories)) / len(categories)
        else:
            scores = scores / scores.sum()
        
        return scores
    
    def analyze(self, image_path: str) -> Dict:
        """
        Complete analysis of product image
        
        Returns:
            Dictionary containing:
            - category: Predicted product category
            - category_confidence: Confidence score for category
            - description: Generated product description
            - alternatives: Top 3 category alternatives with scores
        """
        logger.info(f"Starting analysis for: {image_path}")
        
        # Load image
        image = self._load_image(image_path)
        
        # Predict category
        if self.model_type == "clip":
            categories, scores = self._predict_category_clip(image)
        elif self.model_type == "blip":
            categories, scores = self._predict_category_blip(image)
        else:  # ensemble
            cat_blip, scores_blip = self._predict_category_blip(image)
            cat_clip, scores_clip = self._predict_category_clip(image)
            # Average the scores
            categories = cat_blip
            scores = [(s1 + s2) / 2 for s1, s2 in zip(scores_blip, scores_clip)]
        
        # Generate description
        description = self._generate_description(image)
        
        result = {
            "category": categories[0],
            "category_confidence": float(scores[0]),
            "description": description,
            "alternatives": [
                {"category": cat, "confidence": float(score)}
                for cat, score in zip(categories[1:], scores[1:])
            ],
            "model_used": self.model_type
        }
        
        logger.info(f"Analysis complete. Predicted category: {result['category']}")
        
        return result
    
    def analyze_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple product images
        
        Returns:
            List of analysis results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {e}")
                results.append({"error": str(e), "image": image_path})
        
        return results


if __name__ == "__main__":
    # Example usage
    analyzer = ProductAnalyzer(model_type="blip")
    
    # Analyze a single image
    result = analyzer.analyze("path/to/product/image.jpg")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['category_confidence']:.2%}")
    print(f"Description: {result['description']}")
