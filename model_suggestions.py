from openmed import get_model_suggestions

# Get model suggestions based on text content
text = "Metastatic breast cancer treated with paclitaxel and trastuzumab"
suggestions = get_model_suggestions(text)

print("Model suggestions for the text above:")
for key, info, reason in suggestions:
    print(f"  {info.display_name} -> {reason}")
