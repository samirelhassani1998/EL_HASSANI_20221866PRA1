import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

st.title("Analyse des données Uber - Avril 2014")

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
st.subheader("Premières lignes")
st.write(df.head())

st.subheader("Dernières lignes")
st.write(df.tail())

st.subheader("Nombre de lignes")
st.write(df.shape[0])

st.subheader("Statistiques descriptives")
st.write(df.describe())

st.subheader("Informations sur le DataFrame")
st.write(df.info())

# Créer un histogramme de la distribution du jour du mois
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots()
sns.histplot(df['dom'], bins=30, kde=False, ax=ax)
ax.set_title('Day of the Month Distribution - Uber - April 2014')
ax.set_xlabel('Day of the Month')
ax.set_ylabel('Frequency')

st.subheader("Histogramme de la distribution du jour du mois")
st.pyplot(fig)

import pydeck as pdk

st.title("Carte GeoJSON avec PyDeck et Streamlit")

# Définir l'URL de la source de données
DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"

# Définir les coordonnées du polygone de couverture de terrain
LAND_COVER = [[[-123.0, 49.196], [-123.0, 49.324], [-123.306, 49.324], [-123.306, 49.196]]]

# Définir l'état initial de la vue
INITIAL_VIEW_STATE = pdk.ViewState(latitude=49.254, longitude=-123.13, zoom=11, max_zoom=16, pitch=45, bearing=0)

# Créer une couche pour le polygone de couverture de terrain
polygon = pdk.Layer(
    "PolygonLayer",
    LAND_COVER,
    stroked=False,
    # traite les données comme une paire de longitude-latitude plate
    get_polygon="-",
    get_fill_color=[0, 0, 0, 20],
)

# Créer une couche pour les données géospatiales
geojson = pdk.Layer(
    "GeoJsonLayer",
    DATA_URL,
    opacity=0.8,
    stroked=False,
    filled=True,
    extruded=True,
    wireframe=True,
    get_elevation="properties.valuePerSqm / 20",
    get_fill_color="[255, 255, properties.growth * 255]",
    get_line_color=[255, 255, 255],
)

# Créer une carte contenant les deux couches
r = pdk.Deck(layers=[polygon, geojson], initial_view_state=INITIAL_VIEW_STATE)

# Afficher la carte dans Streamlit
st.pydeck_chart(r)

st.title("Analyse des trajets de taxi à New York")

st.header("Données")

path2 = "https://raw.githubusercontent.com/uber-web/kepler.gl-data/master/nyctrips/data.csv"
df3 = pd.read_csv(path2, delimiter=',')

st.write(df3.head())

st.header("Informations générales sur le DataFrame")
st.write(df3.info())

st.header("Statistiques descriptives pour chaque colonne")
st.write(df3.describe())

st.header("Analyses")

mean_trip_distance = df3['trip_distance'].mean()
st.write(f"Distance moyenne parcourue par trajet : {mean_trip_distance:.2f} km")

mean_tip_amount = df3['tip_amount'].mean()
st.write(f"Montant moyen des pourboires : {mean_tip_amount:.2f} $")

mean_passenger_count = df3['passenger_count'].mean()
st.write(f"Moyenne du nombre de passagers par trajet : {mean_passenger_count:.2f}")

mean_total_amount = df3['total_amount'].mean()
st.write(f"Montant moyen total des trajets : {mean_total_amount:.2f} $")

passenger_count_counts = df3['passenger_count'].value_counts()
st.write("Nombre de trajets par nombre de passagers :")
st.write(passenger_count_counts)

st.header("Visualisations")

st.subheader("Histogramme des distances de trajet")
plt.figure()
sns.histplot(df3['trip_distance'], bins=50)
plt.title("Histogramme des distances de trajet")
plt.xlabel("Distance (km)")
plt.ylabel("Nombre de trajets")
st.pyplot(plt)

st.subheader("Boxplot des montants totaux des trajets")
plt.figure()
sns.boxplot(x=df3['total_amount'])
plt.title("Boxplot des montants totaux des trajets")
plt.xlabel("Montant total ($)")
st.pyplot(plt)

st.subheader("Nombre de trajets par nombre de passagers")
plt.figure()
sns.barplot(x=passenger_count_counts.index, y=passenger_count_counts.values)
plt.title("Nombre de trajets par nombre de passagers")
plt.xlabel("Nombre de passagers")
plt.ylabel("Nombre de trajets")
st.pyplot(plt)

# Fonction pour créer la heatmap
def create_heatmap(df):
    pickup_heatmap_data = df.groupby([
        df['pickup_latitude'].round(2),
        df['pickup_longitude'].round(2)
    ]).size().unstack().fillna(0)

    sns.heatmap(pickup_heatmap_data, cmap="viridis")
    plt.title("Densité des points de prise en charge")
    plt.xlabel("Longitude arrondie")
    plt.ylabel("Latitude arrondie")
    return plt

# Titre de l'application Streamlit
st.title("Heatmap de la densité des points de prise en charge")

# Affichez la heatmap dans l'application Streamlit
st.pyplot(create_heatmap(df3))

# Fonction pour créer le scatterplot
def create_scatterplot(df):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=df['pickup_latitude'].mean(),
        longitude=df['pickup_longitude'].mean(),
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["pickup_longitude", "pickup_latitude"],
        get_radius=50,
        get_fill_color=[180, 0, 200, 140],
        pickable=True,
        auto_highlight=True
    )

    deck = pdk.Deck(layers=[scatterplot_layer], initial_view_state=INITIAL_VIEW_STATE)
    return deck

# Titre de l'application Streamlit
st.title("Scatterplot des points de prise en charge")

# Affichez le scatterplot dans l'application Streamlit
st.pydeck_chart(create_scatterplot(df3))

# Fonction pour créer la heatmap
def create_heatmap(df):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=df['pickup_latitude'].mean(),
        longitude=df['pickup_longitude'].mean(),
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["pickup_longitude", "pickup_latitude"],
        opacity=0.9,
        pickable=True,
        auto_highlight=True
    )

    deck_heatmap = pdk.Deck(layers=[heatmap_layer], initial_view_state=INITIAL_VIEW_STATE)
    return deck_heatmap

# Titre de l'application Streamlit
st.title("Heatmap des points de prise en charge")

# Affichez la heatmap dans l'application Streamlit
st.pydeck_chart(create_heatmap(df3))
