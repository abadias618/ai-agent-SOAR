import pandas as pd

df = pd.read_json("./data/train.json")
print(df.head())
print("\n\n\n")
print(df)
print("\n\n\n")
print(df[0])
print("\n\n\n")
print(df[0][0])