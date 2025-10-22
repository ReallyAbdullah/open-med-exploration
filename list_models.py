from openmed import list_models

# See all available models
models = list_models()
print(f"Total models available: {len(models)}")
print("\nFirst 10 models:")
for model in models[:10]:
    print(f"  - {model}")

from openmed.core.model_registry import list_model_categories
# List all model categories
categories = list_model_categories()
print("Available model categories:")
for category in categories:
    print(f"  - {category}")

from openmed.core.model_registry import get_models_by_category
# Get models by category
disease_models = get_models_by_category("Disease")
print("Disease detection models:")
for model in disease_models:
    print(f"  - {model.display_name} ({model.size_category})")