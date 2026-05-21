import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/apple/Desktop/titanic_clean.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Chart 1 — survival by sex
df.groupby("Sex")["Survived"].mean().plot(kind="bar", ax=axes[0,0], color=["#4A90D9","#E87C6A"])
axes[0,0].set_title("Survival rate by gender")
axes[0,0].set_ylabel("Survival rate")

# Chart 2 — age distribution
sns.histplot(data=df, x="Age", hue="Survived", bins=25, ax=axes[0,1])
axes[0,1].set_title("Age distribution by survival")

# Chart 3 — survival by class
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[1,0], color="#5BA85C")
axes[1,0].set_title("Survival rate by class")

# Chart 4 — correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".1f", ax=axes[1,1], cmap="coolwarm")
axes[1,1].set_title("Correlation heatmap")

plt.tight_layout()
plt.savefig("/Users/apple/Desktop/titanic_clean.png")
plt.show()