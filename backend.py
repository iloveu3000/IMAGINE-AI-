# solid_mechanics_ai.py
import torch
import torch.nn as nn
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF processing
import json

class SolidMechanicsAI:
    def __init__(self):
        self.qa_pipeline = None
        self.vectorizer = None
        self.corpus = []
        self.qa_pairs = []
        self.setup_models()
        self.load_pdf_content()
        
    def setup_models(self):
        """Initialize the AI models"""
        try:
            # Use a smaller, more efficient model for this domain
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
        except:
            # Fallback to rule-based system if model fails to load
            self.qa_pipeline = None
            
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def load_pdf_content(self):
        """Extract and process content from the solid mechanics PDF"""
        pdf_content = """
        TORSION:
        - Torsion formula: T = (Ï€/16) Ã— Ï„ Ã— dÂ³
        - For solid shafts: T = (Ï€/16) Ã— Ï„ Ã— dÂ³
        - For hollow shafts: T = (Ï€/16) Ã— Ï„ Ã— (dâ‚€â´ - dáµ¢â´)/dâ‚€
        - Shear stress: Ï„ = (16T)/(Ï€dÂ³)
        - Polar modulus = Polar moment of inertia / Radius
        
        SPRINGS:
        - Spring deflection: Î´ = (64WRÂ³n)/(Gdâ´)
        - Spring stiffness: k = W/Î´ = (Gdâ´)/(64RÂ³n)
        - Energy stored: U = (1/2)WÎ´
        - Shear stress in spring: Ï„ = (16WR)/(Ï€dÂ³)
        
        PRESSURE VESSELS:
        - Circumferential stress: Ïƒ_c = (Pd)/(2t)
        - Longitudinal stress: Ïƒ_L = (Pd)/(4t)
        - Thin-walled assumption: t << d
        
        PRINCIPAL STRESSES:
        - Principal stresses: Ïƒâ‚,â‚‚ = (Ïƒ_x + Ïƒ_y)/2 Â± âˆš[((Ïƒ_x - Ïƒ_y)/2)Â² + Ï„_xyÂ²]
        - Maximum shear stress: Ï„_max = âˆš[((Ïƒ_x - Ïƒ_y)/2)Â² + Ï„_xyÂ²]
        - Angle: tan(2Î¸) = (2Ï„_xy)/(Ïƒ_x - Ïƒ_y)
        
        STRAIN ENERGY:
        - Strain energy in tension: U = (ÏƒÂ²V)/(2E)
        - Volumetric strain: Î”V/V = (Ïƒ_x + Ïƒ_y + Ïƒ_z)(1-2Î½)/E
        
        MATERIAL PROPERTIES:
        - Young's modulus (E), Shear modulus (G), Poisson's ratio (Î½)
        - Relationship: G = E/(2(1+Î½))
        
        ASSUMPTIONS IN TORSION:
        1. The shaft length is uniform
        2. Circular shafts remain circular after twisting
        3. Material is uniform throughout
        4. Twist along shaft is uniform
        
        EXAMPLES:
        - For Ï„=46 N/mmÂ², d=250mm: T=(Ï€/16)Ã—46Ã—(250)Â³=141.05 N/mm
        - Spring with W=200N, R=60mm, n=10, d=10mm, G=80Ã—10Â³ N/mmÂ²:
          Î´=(64Ã—200Ã—60Â³Ã—10)/(80Ã—10Â³Ã—10â´)=34.56mm
          k=200/34.56=5.78 N/mm
          U=0.5Ã—200Ã—34.56=3456 NÂ·mm
        """
        
        # Process the content into QA pairs
        self.process_content_to_qa(pdf_content)
        
    def process_content_to_qa(self, content):
        """Convert PDF content into question-answer pairs"""
        lines = content.split('\n')
        current_topic = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(':'):
                current_topic = line[:-1]
            elif ':' in line and '=' in line:
                # Formula lines
                parts = line.split(':', 1)
                if len(parts) == 2:
                    concept = parts[0].strip()
                    formula = parts[1].strip()
                    self.qa_pairs.append({
                        'question': f'What is the formula for {concept}?',
                        'answer': f'The formula for {concept} is: {formula}'
                    })
            elif '-' in line and len(line) > 10:
                # Concept lines
                concept = line[1:].strip()
                self.qa_pairs.append({
                    'question': f'Explain {concept}',
                    'answer': f'{concept} is an important concept in solid mechanics. {self.generate_explanation(concept)}'
                })
        
        # Add the corpus for similarity search
        self.corpus = [pair['answer'] for pair in self.qa_pairs]
        if self.corpus:
            self.vectorizer.fit(self.corpus)
    
    def generate_explanation(self, concept):
        """Generate explanations for concepts"""
        explanations = {
            'torsion': 'Torsion refers to the twisting of an object due to an applied torque. It is crucial in shaft design.',
            'springs': 'Springs are mechanical components that store and release energy, commonly used for shock absorption.',
            'pressure vessels': 'Pressure vessels are containers designed to hold gases or liquids at high pressures.',
            'principal stresses': 'Principal stresses are the maximum and minimum normal stresses at a point.',
            'strain energy': 'Strain energy is the energy stored in a material when it is deformed elastically.'
        }
        
        for key, explanation in explanations.items():
            if key in concept.lower():
                return explanation
        return "This is a fundamental concept in solid mechanics engineering."
    
    def find_similar_question(self, question):
        """Find the most similar question using TF-IDF"""
        if not self.corpus:
            return None
            
        question_vec = self.vectorizer.transform([question])
        corpus_vec = self.vectorizer.transform(self.corpus)
        
        similarities = cosine_similarity(question_vec, corpus_vec)[0]
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] > 0.3:  # Threshold
            return self.qa_pairs[best_match_idx]
        return None
    
    def ask_question(self, question):
        """Answer questions about solid mechanics"""
        # First try to find similar question in our knowledge base
        similar_qa = self.find_similar_question(question)
        if similar_qa:
            return similar_qa['answer']
        
        # If no good match, use the QA pipeline
        if self.qa_pipeline:
            try:
                # Create context from our knowledge
                context = " ".join([qa['answer'] for qa in self.qa_pairs[:5]])
                result = self.qa_pipeline(question=question, context=context)
                return result['answer']
            except:
                pass
        
        # Fallback response
        return "I'm trained in solid mechanics. I can help with torsion, springs, pressure vessels, stress analysis, and material properties. Could you please rephrase your question?"
    
    def calculate_formula(self, formula_name, **kwargs):
        """Calculate engineering formulas"""
        calculations = {
            'torsion_solid': lambda: (np.pi/16) * kwargs.get('tau', 0) * (kwargs.get('d', 0)**3),
            'torsion_hollow': lambda: (np.pi/16) * kwargs.get('tau', 0) * ((kwargs.get('d0', 0)**4 - kwargs.get('di', 0)**4) / kwargs.get('d0', 1)),
            'spring_deflection': lambda: (64 * kwargs.get('W', 0) * (kwargs.get('R', 0)**3) * kwargs.get('n', 0)) / (kwargs.get('G', 1) * (kwargs.get('d', 1)**4)),
            'circumferential_stress': lambda: (kwargs.get('P', 0) * kwargs.get('d', 0)) / (2 * kwargs.get('t', 1)),
            'longitudinal_stress': lambda: (kwargs.get('P', 0) * kwargs.get('d', 0)) / (4 * kwargs.get('t', 1)),
            'principal_stress_max': lambda: (kwargs.get('sigma_x', 0) + kwargs.get('sigma_y', 0))/2 + np.sqrt(((kwargs.get('sigma_x', 0) - kwargs.get('sigma_y', 0))/2)**2 + kwargs.get('tau_xy', 0)**2),
            'principal_stress_min': lambda: (kwargs.get('sigma_x', 0) + kwargs.get('sigma_y', 0))/2 - np.sqrt(((kwargs.get('sigma_x', 0) - kwargs.get('sigma_y', 0))/2)**2 + kwargs.get('tau_xy', 0)**2)
        }
        
        if formula_name in calculations:
            try:
                result = calculations[formula_name]()
                return f"The calculated result is: {result:.4f}"
            except:
                return "Error in calculation. Please check the input parameters."
        else:
            return "Formula not recognized."

# Flask API for web integration
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ai_model = SolidMechanicsAI()

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    answer = ai_model.ask_question(question)
    return jsonify({'question': question, 'answer': answer})

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    formula = data.get('formula', '')
    parameters = data.get('parameters', {})
    
    if not formula:
        return jsonify({'error': 'No formula specified'}), 400
    
    result = ai_model.calculate_formula(formula, **parameters)
    return jsonify({'formula': formula, 'result': result})

@app.route('/api/topics', methods=['GET'])
def get_topics():
    topics = [
        "Torsion and Shaft Design",
        "Springs and Deflection",
        "Pressure Vessels",
        "Principal Stresses",
        "Strain Energy",
        "Material Properties"
    ]
    return jsonify({'topics': topics})

if __name__ == '__main__':
    print("Solid Mechanics AI Assistant Starting...")
    print("Available endpoints:")
    print("  POST /api/ask - Ask questions about solid mechanics")
    print("  POST /api/calculate - Perform engineering calculations")
    print("  GET /api/topics - Get available topics")
    
    app.run(debug=True, port=5000)