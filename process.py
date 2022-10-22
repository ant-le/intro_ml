
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DIR = Path(__file__).parent

df1 = pd.read_csv(DIR / "data" / "winequality-red.csv", sep=";")
df1.loc[:, "type"] = "red"

df2 = pd.read_csv(DIR / "data" / "winequality-white.csv", sep=";")
df2.loc[:, "type"] = "white"

wine = pd.concat([df1, df2])

# characteristics of dataset
print("Number of Samples: " + str(wine.shape[0]) + ", Number of Attributes: " + str(wine.shape[1]))
print(wine.dtypes)

# frequency table for target attribute
print(wine.quality.value_counts().sort_index().to_latex())

# predictor attributes
wine.type.value_counts() # binary type (nominal)

with plt.style.context("seaborn-white"): # alcohol might be transformable
    plt.figure()
    plt.hist(wine["citric acid"], alpha=.8, color="lightgray", edgecolor="black", bins=20)
    plt.xlabel("Citric Acid")
    plt.ylabel("Frequency")
    plt.savefig(DIR / "images" / "acid.pdf")


with plt.style.context("seaborn-white"): # alcohol might be transformable
    plt.figure()
    plt.hist(wine.density, alpha=.8, color="lightgray", edgecolor="black", bins=40)
    plt.xlabel("Density")
    plt.ylabel("Frequency")
    plt.savefig(DIR / "images" / "density.pdf")
    plt.show()