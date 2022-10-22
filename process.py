
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent / "data"

df1 = pd.read_csv(DATA_PATH.joinpath("winequality-red.csv"), sep=";")
df1.loc[:, "type"] = "red"

df2 = pd.read_csv(DATA_PATH.joinpath("winequality-white.csv"), sep=";")
df2.loc[:, "type"] = "white"

wine = pd.concat([df1, df2])

# characteristics of dataset
print("Number of Samples: " + str(wine.shape[0]) + ", Number of Attributes: " + str(wine.shape[1]))
print(wine.dtypes)

# target attribute
print(wine.quality.value_counts().sort_index().to_latex())

# predictor attributes
wine.type.value_counts() # binary type (nominal)

with plt.style.context("seaborn-white"): # alcohol might be transformable
    plt.hist(wine.alcohol, alpha=.8, color="lightgray", edgecolor="black", bins=15)
    plt.xlabel("Alcohol")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig


with plt.style.context("seaborn-white"): # alcohol might be transformable
    plt.hist(wine.density, alpha=.8, color="lightgray", edgecolor="black", bins=40)
    plt.xlabel("Density")
    plt.ylabel("Frequency")
    plt.show()
    

i=4
plt.hist(wine.iloc[:,i], bins=20)
plt.title("Histogram for " + wine.iloc[:,i].name)
plt.show()