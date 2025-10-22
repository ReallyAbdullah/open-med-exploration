# Patient Analysis Report

## Clinical Intake
### Symptoms
['bloating', 'abdominal pain']

### Onset Weeks
12

### Current Meds
['omeprazole']

### Context Notes
anxiety present

## Extracted Entities
- omeprazole (chem)
- peppermint oil (chem)
- abdominal (anatomy)
- stools (anatomy)
- gut (anatomy)
- FODMAP (chem)
- peppermint (chem)
- IBS (disease)

## Treatment Plan
### Variant
calm_breathing

### Rationale
Adaptive selection for GI + anxiety co-management.

### Steps
['5m diaphragmatic breathing', '10m gut-directed relaxation', '3m reflection']

### Safety Flags
['Monitor reflux; avoid menthol-heavy supplements if reflux worsens.']

## FHIR Bundle Preview
```json
{
  "entries": [
    {
      "resourceType": "Condition",
      "code": {
        "coding": [
          {
            "system": "http://snomed.info/sct",
            "code": "10743008",
            "display": "IBS"
          }
        ]
      }
    },
    {
      "resourceType": "DocumentReference",
      "content": [
        {
          "attachment": {
            "contentType": "text/markdown",
            "data": "# Summary\n## Entities\n- **omeprazole** (chem, conf=0.96)\n- **peppermint oil** (chem, conf=0.94)\n- **abdominal** (anatomy, conf=0.75)\n- **stools** (anatomy, conf=0.67)\n- **gut** (anatomy, conf=0.88)\n- **FODMAP** (chem, conf=0.90)\n- **peppermint** (chem, conf=0.85)\n- **IBS** (disease, conf=0.60)\n## Variant\n- `calm_breathing`\n- Adaptive selection for GI + anxiety co-management.\n- Steps:\n  - 5m diaphragmatic breathing\n  - 10m gut-directed relaxation\n  - 3m reflection\n## Safety\n- Monitor reflux; avoid menthol-heavy supplements if reflux worsens."
          }
        }
      ]
    }
  ]
}
```
