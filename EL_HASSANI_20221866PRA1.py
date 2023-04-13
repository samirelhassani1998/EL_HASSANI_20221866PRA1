import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# Télécharger le fichier CSV à partir de Google Drive
url = "https://drive.google.com/uc?id=1qoK_zLLWWPjY3HaPN4WXH2q1OayJiQtz"
gdown.download(url, "uber-raw-data-apr14.csv", quiet=False)

# Charger les données
df = pd.read_csv("uber-raw-data-apr14.csv", delimiter=',')
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Fonctions pour extraire les informations pertinentes
def get_dom(dt):
    return dt.day

def get_weekday(dt):
    return dt.weekday()

def get_hour(dt):
    return dt.hour

# Appliquer les fonctions pour extraire les informations pertinentes
df['dom'] = df['Date/Time'].map(get_dom)
df['weekday'] = df['Date/Time'].map(get_weekday)
df['hour'] = df['Date/Time'].map(get_hour)

# Afficher les premières lignes, les dernières lignes et les informations du DataFrame
print(df.head())
print(df.tail())
print(df.shape[0])
print(df.describe())
print(df.info())

# Créer un histogramme de la distribution du jour du mois
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.histplot(df['dom'], bins=30, kde=False)
plt.title('Day of the Month Distribution - Uber - April 2014')
plt.xlabel('Day of the Month')
plt.ylabel('Frequency')
plt.show()
