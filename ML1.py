import pandas as pd
import numpy as np

def find_s(data):
    # Initialize the most specific hypothesis
    hypothesis = None
    
    for _, row in data.iterrows():
        if row.iloc[-1] == 'Yes': # Positive instance
            if hypothesis is None:
                hypothesis = list(row.iloc[:-1])
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row.iloc[i]:
                        hypothesis[i] = '?'
        print(f"Current Row: {list(row)}, Hypothesis: {hypothesis}")
    return hypothesis

# Sample Dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Same'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\nRunning Find-S Algorithm...")
final_hypothesis = find_s(df)
print("\nFinal Specific Hypothesis:", final_hypothesis)
