import pandas as pd

# Create a DataFrame with your dataset
data = {
    'Fruit': ['apple', 'mango', 'apple', 'orange'],
    'CategoricalValue': [1, 2, 1, 3],
    'Price': [5, 10, 15, 20]
}
df = pd.DataFrame(data)

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Fruit'], prefix=[''])

print(df_encoded)
