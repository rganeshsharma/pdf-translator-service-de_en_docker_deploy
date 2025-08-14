#!/usr/bin/env python3
"""Test script for downloaded model"""

import sys
from pathlib import Path

try:
    from transformers import MarianMTModel, MarianTokenizer
    
    model_path = "./models/Helsinki-NLP/opus-mt-de-en"
    
    print("🧪 Testing downloaded model...")
    print(f"📁 Model path: {Path(model_path).absolute()}")
    
    # Load model and tokenizer from local files
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    
    # Test translation
    test_text = "Hallo Welt! Wie geht es dir?"
    print(f"🇩🇪 German: {test_text}")
    
    tokens = tokenizer(test_text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    print(f"🇺🇸 English: {result}")
    print("✅ Model test successful!")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    sys.exit(1)
