import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import spearmanr

data = """
             Study  Tokens (B)   Neuro papers
English       N/A    2733        Well studied
French          I     318        Well studied
Tamil           I       3.4      Understudied       
Spanish         I     433        Well studied       
Turkish         I      71        Understudied     
Vietnamese      I     116        Understudied     
Marathi         I      14        Understudied     
Afrikaans       I       1.7      Understudied       
Dutch           I      73        Well studied     
Norwegian       I      27        Somewhat studied     
Farsi           I      52        Understudied     
Romanian        I      52        Understudied     
Lithuanian      I      11        Understudied     
Hindi          II      24        Somewhat studied     
Korean         II      26        Well studied      
Arabic         II      57        Somewhat studied      
Polish         II     130        Somewhat studied       
Russian        II     713        Somewhat studied        
Portuguese     II     146        Somewhat studied           
Mandarin       II      39        Well studied        
German         II     347        Well studied             
Italian        II     162        Somewhat studied     
"""


df = pd.read_csv(StringIO(data), sep=r"\s{2,}", engine="python")
df.columns = ["Study", "Tokens (B)", "Neuro papers"]
print(df)

df["Tokens (B)"] = pd.to_numeric(df["Tokens (B)"])
df = df.dropna(subset=["Study"])

order = ["Well studied", "Somewhat studied", "Understudied"]
df["Neuro papers"] = pd.Categorical(df["Neuro papers"], categories=order, ordered=True)
df = df.sort_values("Neuro papers")

groups = [df[df["Neuro papers"] == category]["Tokens (B)"] for category in order]
anova_stat, anova_p = f_oneway(*groups)
print(f"ANOVA: F = {anova_stat:.2f}, p = {anova_p:.4f}") # ANOVA: F = 2.00, p = 0.1647

order = {"Understudied": 1, "Somewhat studied": 2, "Well studied": 3}
df["Ordinal Neuro papers"] = df["Neuro papers"].map(order)
stat, p_value = spearmanr(df["Ordinal Neuro papers"], df["Tokens (B)"])
print(f"Spearman's Rank Correlation: Rho = {stat:.2f}, p = {p_value:.4f}") # Spearman's Rank Correlation: Rho = 0.50, p = 0.0201

plt.figure(figsize=(5*.7, 3*.7), dpi=300)
sns.boxplot(x="Neuro papers", y="Tokens (B)", data=df, linewidth=1, fill=True, linecolor="k")
plt.xlabel("Coverage in fMRI studies")
plt.ylabel("mC4 tokens (B)")
new_labels = ["Well\nstudied", "Somewhat\nstudied", "Understudied"]
plt.gca().set_xticklabels(new_labels)
plt.tight_layout()
plt.show()