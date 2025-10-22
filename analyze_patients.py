# Set up logging suppression for external libraries
import logging
import warnings
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import torch

# Suppress external library logs BEFORE importing OpenMed
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up OpenMed logging
from openmed.utils import setup_logging
setup_logging(level="WARNING")

from openmed import analyze_text

# Configure device for model inference
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

def analyze_patient_data(description: str) -> Tuple[Dict, Dict]:
    """
    Analyze a patient description using both disease and pharmaceutical models.
    Returns tuple of (disease_entities, pharma_entities)
    """
    # Analyze with disease model
    disease_result = analyze_text(
        description,
        model_name="disease_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device
    )
    
    # Analyze with pharmaceutical model
    pharma_result = analyze_text(
        description,
        model_name="pharma_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device
    )
    
    return disease_result, pharma_result

def analyze_entity_patterns(disease_entities: List, pharma_entities: List) -> Dict:
    """Analyze patterns and relationships between entities"""
    patterns = {
        'disease_pharma_pairs': defaultdict(list),
        'high_confidence_entities': [],
        'low_confidence_entities': []
    }
    
    # Group diseases with related pharmaceuticals
    disease_spans = {(e.start, e.end): e for e in disease_entities}
    pharma_spans = {(e.start, e.end): e for e in pharma_entities}
    
    for disease in disease_entities:
        for pharma in pharma_entities:
            # Check if entities are close to each other (within 50 characters)
            if abs(disease.start - pharma.start) <= 50:
                patterns['disease_pharma_pairs'][disease.text].append(pharma.text)
    
    # Identify high and low confidence entities
    CONFIDENCE_THRESHOLD = 0.7
    for entity in disease_entities + pharma_entities:
        if entity.confidence >= CONFIDENCE_THRESHOLD:
            patterns['high_confidence_entities'].append((entity.text, entity.label, entity.confidence))
        else:
            patterns['low_confidence_entities'].append((entity.text, entity.label, entity.confidence))
            
    return patterns

def main():
    # Load the dataset
    print("Loading NCBI Open-Patients dataset...")
    df = pd.read_json("hf://datasets/ncbi/Open-Patients/Open-Patients.jsonl", lines=True)
    print(f"Dataset loaded with {len(df)} records\n")
    
    # Initialize data collection structures
    stats = {
        'disease_entity_types': Counter(),
        'pharma_entity_types': Counter(),
        'all_confidences_disease': [],
        'all_confidences_pharma': [],
        'disease_pharma_patterns': defaultdict(list),
        'common_disease_combos': Counter(),
        'common_drug_combos': Counter()
    }
    
    # Process a sample of records
    sample_size = min(20, len(df))  # Increased sample size for better pattern detection
    sample_df = df.sample(n=sample_size, random_state=42)
    
    print(f"Analyzing {sample_size} patient descriptions...")
    
    # Analyze each patient description
    for idx, row in sample_df.iterrows():
        print(f"\nAnalyzing record {idx + 1}/{sample_size}")
        disease_result, pharma_result = analyze_patient_data(row['description'])
        
        # Collect entities and analyze patterns
        patterns = analyze_entity_patterns(disease_result.entities, pharma_result.entities)
        
        # Update entity type counts
        for entity in disease_result.entities:
            stats['disease_entity_types'][entity.label] += 1
            stats['all_confidences_disease'].append(entity.confidence)
            
        for entity in pharma_result.entities:
            stats['pharma_entity_types'][entity.label] += 1
            stats['all_confidences_pharma'].append(entity.confidence)
            
        # Update disease-drug patterns
        for disease, drugs in patterns['disease_pharma_pairs'].items():
            stats['disease_pharma_patterns'][disease].extend(drugs)
        
        # Track disease and drug combinations within each record
        diseases = [e.text for e in disease_result.entities]
        drugs = [e.text for e in pharma_result.entities]
        
        # Record co-occurring diseases and drugs
        for i in range(len(diseases)):
            for j in range(i+1, len(diseases)):
                combo = tuple(sorted([diseases[i], diseases[j]]))
                stats['common_disease_combos'][combo] += 1
                
        for i in range(len(drugs)):
            for j in range(i+1, len(drugs)):
                combo = tuple(sorted([drugs[i], drugs[j]]))
                stats['common_drug_combos'][combo] += 1
    
    # Print comprehensive analysis results
    print("\n=== Analysis Results ===")
    print(f"Processed {sample_size} patient descriptions\n")
    
    print("1. Entity Type Distribution")
    print("-" * 50)
    print("\nDisease Entity Types:")
    for entity_type, count in stats['disease_entity_types'].most_common():
        print(f"  {entity_type}: {count}")
    
    print("\nPharmaceutical Entity Types:")
    for entity_type, count in stats['pharma_entity_types'].most_common():
        print(f"  {entity_type}: {count}")
    
    # Calculate and print model performance metrics
    avg_confidence_disease = (sum(stats['all_confidences_disease']) / 
                            len(stats['all_confidences_disease'])) if stats['all_confidences_disease'] else 0
    avg_confidence_pharma = (sum(stats['all_confidences_pharma']) / 
                           len(stats['all_confidences_pharma'])) if stats['all_confidences_pharma'] else 0
    
    print("\n2. Model Performance Metrics")
    print("-" * 50)
    print(f"Disease Detection Average Confidence: {avg_confidence_disease:.3f}")
    print(f"Pharmaceutical Detection Average Confidence: {avg_confidence_pharma:.3f}")
    
    print("\n3. Common Disease-Drug Associations")
    print("-" * 50)
    for disease, drugs in stats['disease_pharma_patterns'].items():
        if drugs:  # Only show diseases with associated drugs
            print(f"\nDisease: {disease}")
            drug_counts = Counter(drugs)
            for drug, count in drug_counts.most_common(3):  # Show top 3 associated drugs
                print(f"  - {drug} (mentioned {count} times)")
    
    print("\n4. Common Disease Co-occurrences")
    print("-" * 50)
    for (disease1, disease2), count in stats['common_disease_combos'].most_common(5):
        print(f"  {disease1} + {disease2}: {count} occurrences")
    
    print("\n5. Common Drug Co-prescriptions")
    print("-" * 50)
    for (drug1, drug2), count in stats['common_drug_combos'].most_common(5):
        print(f"  {drug1} + {drug2}: {count} occurrences")

if __name__ == "__main__":
    main()