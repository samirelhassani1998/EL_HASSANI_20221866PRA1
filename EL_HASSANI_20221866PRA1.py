import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import gdown
import os

# Titre de l'application Streamlit
st.title("Analyse des trajets de taxi à New York")

# Fonction pour charger les données
@st.cache
def load_data():
    path = "https://raw.githubusercontent.com/uber-web/kepler.gl-data/master/nyctrips/data.csv"
    df = pd.read_csv(path, delimiter=',')
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    return df

# Chargement des données
df = load_data()

# Section Informations générales sur le DataFrame
st.header("Informations générales sur le DataFrame")
st.write(df.info())

# Section Statistiques descriptives pour chaque colonne
st.header("Statistiques descriptives pour chaque colonne")
st.write(df.describe())

# Section Analyses

# Moyenne de la distance parcourue par trajet
mean_trip_distance = df['trip_distance'].mean()

# Moyenne du montant des pourboires
mean_tip_amount = df['tip_amount'].mean()

# Moyenne du nombre de passagers par trajet
mean_passenger_count = df['passenger_count'].mean()

# Moyenne du montant total des trajets
mean_total_amount = df['total_amount'].mean()

# Nombre de trajets par nombre de passagers
passenger_count_counts = df['passenger_count'].value_counts()

# Section Affichage des résultats des analyses
st.header("Résultats des analyses")

# Affichage de la distance moyenne parcourue par trajet
st.write(f"Distance moyenne parcourue par trajet : {mean_trip_distance:.2f} km")

# Affichage du montant moyen des pourboires
st.write(f"Montant moyen des pourboires : {mean_tip_amount:.2f} $")

# Affichage de la moyenne du nombre de passagers par trajet
st.write(f"Moyenne du nombre de passagers par trajet : {mean_passenger_count:.2f}")

# Affichage du montant moyen total des trajets
st.write(f"Montant moyen total des trajets : {mean_total_amount:.2f} $")

# Affichage du nombre de trajets par nombre de passagers
st.write("Nombre de trajets par nombre de passagers :")
st.write(passenger_count_counts)

# Section Visualisations

# Histogramme des distances de trajet
st.subheader("Histogramme des distances de trajet")
bins = st.slider('Nombre de barres', 1, 100, 50)
plt.figure()
sns.histplot(df['trip_distance'], bins=bins)
plt.title("Histogramme des distances de trajet")
plt.xlabel("Distance (km)")
plt.ylabel("Nombre de trajets")
st.pyplot(plt)

# Boxplot des montants totaux des trajets
st.subheader("Boxplot des montants totaux des trajets")
plt.figure()
sns.boxplot(x=df['total_amount'])
plt.title("Boxplot des montants totaux des trajets")
plt.xlabel("Montant total ($)")


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

# Create a scatterplot layer for pickup points
scatterplot_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df3,
    get_position=["pickup_longitude", "pickup_latitude"],
    get_radius=50,
    get_fill_color=[255, 0, 0, 140],
    pickable=True,
    auto_highlight=True
)

# Create a scatterplot layer for dropoff points
scatterplot_dropoff_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df3,
    get_position=["dropoff_longitude", "dropoff_latitude"],
    get_radius=50,
    get_fill_color=[0, 255, 0, 140],
    pickable=True,
    auto_highlight=True
)

# Create a deck containing both pickup and dropoff scatterplot layers
deck_pickup_dropoff = pdk.Deck(
    layers=[scatterplot_layer, scatterplot_dropoff_layer],
    initial_view_state=pdk.ViewState(
        latitude=df3["pickup_latitude"].mean(),
        longitude=df3["pickup_longitude"].mean(),
        zoom=10,
        pitch=0,
        bearing=0
    )
)

# Render the PyDeck deck in Streamlit
st.pydeck_chart(deck_pickup_dropoff)

# Convert pickup datetime to pandas datetime format
df3['tpep_pickup_datetime'] = pd.to_datetime(df3['tpep_pickup_datetime'])

# Extract hour from pickup datetime
df3['pickup_hour'] = df3['tpep_pickup_datetime'].dt.hour

# Plot the number of trips per hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df3, x='pickup_hour')
plt.title("Nombre de trajets par heure")
plt.xlabel("Heure")
plt.ylabel("Nombre de trajets")

# Render the matplotlib plot in Streamlit
st.pyplot()

# Group data by hour and calculate total revenue
hourly_revenue = df3.groupby('pickup_hour')['total_amount'].sum().reset_index()

# Plot total revenue per hour
plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_revenue, x='pickup_hour', y='total_amount')
plt.title("Revenus totaux par heure")
plt.xlabel("Heure")
plt.ylabel("Revenus totaux")

# Render the matplotlib plot in Streamlit
st.pyplot()

# Create a heatmap layer for dropoff points
heatmap_dropoff = pdk.Layer(
    "HeatmapLayer",
    data=df3,
    opacity=0.8,
    get_position=["dropoff_longitude", "dropoff_latitude"],
    aggregation='"SUM"',
    get_weight="passenger_count"
)

# Create a deck containing the heatmap layer
r = pdk.Deck(
    layers=[heatmap_dropoff],
    initial_view_state=pdk.ViewState(
        latitude=df3["dropoff_latitude"].mean(),
        longitude=df3["dropoff_longitude"].mean(),
        zoom=10,
        pitch=0,
        bearing=0
    )
)

# Render the PyDeck deck in Streamlit
st.pydeck_chart(r)

# Calculate the correlation matrix
corr_matrix = df3[['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de corrélation des caractéristiques numériques')

# Render the matplotlib plot in Streamlit
st.pyplot()

# Calculate the correlation matrix for all numerical columns
correlation_matrix = df3.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de corrélation')

# Render the matplotlib plot in Streamlit
st.pyplot()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df3 = pd.read_csv("path/to/your/data.csv")

# Convert datetime columns to datetime objects
df3['tpep_pickup_datetime'] = pd.to_datetime(df3['tpep_pickup_datetime'])
df3['tpep_dropoff_datetime'] = pd.to_datetime(df3['tpep_dropoff_datetime'])

# Calculate trip duration and convert it to minutes
df3['trip_duration'] = (df3['tpep_dropoff_datetime'] - df3['tpep_pickup_datetime']).dt.total_seconds() / 60

# Extract features and target variable
X = df3[['trip_distance', 'trip_duration']]
y = df3['total_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation metrics
st.write(f'R² score: {r2:.2f}')
st.write(f'RMSE: {rmse:.2f}')
st.write(f'MAE: {mae:.2f}')

# Visualize the predictions vs the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valeurs réelles ($)')
plt.ylabel('Prédictions ($)')
plt.title('Montant total : Prédictions vs Valeurs réelles')
# Add a reference line
max_value = np.maximum(y_test.max(), y_pred.max())
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--')

# Render the matplotlib plot in Streamlit
st.pyplot()

# Convert datetime columns to datetime objects
df3['tpep_pickup_datetime'] = pd.to_datetime(df3['tpep_pickup_datetime'])
df3['tpep_dropoff_datetime'] = pd.to_datetime(df3['tpep_dropoff_datetime'])

# Create new columns for the hour, day of the week, and month
df3['pickup_hour'] = df3['tpep_pickup_datetime'].dt.hour
df3['pickup_day'] = df3['tpep_pickup_datetime'].dt.dayofweek
df3['pickup_month'] = df3['tpep_pickup_datetime'].dt.month

# Plot boxplots of total_amount vs each time column
time_columns = ['pickup_hour', 'pickup_day', 'pickup_month']
for col in time_columns:
    plt.figure()
    sns.boxplot(x=col, y='total_amount', data=df3)
    plt.title(f"Variations du total_amount en fonction de {col}")

    # Render the matplotlib plot in Streamlit
    st.pyplot()
    
# Plot a scatterplot of trip_distance vs total_amount
plt.figure()
sns.scatterplot(x='trip_distance', y='total_amount', data=df3)
plt.title("Relation entre trip_distance et total_amount")

# Render the matplotlib plot in Streamlit
st.pyplot()

# Calculate the correlation matrix of selected columns
corr_columns = ['trip_distance', 'passenger_count', 'fare_amount', 'tip_amount', 'total_amount']
corr_matrix = df3[corr_columns].corr()

# Display the correlation matrix
st.write("Matrice de corrélation:\n", corr_matrix)


# Plot a scatterplot of pickup_longitude vs pickup_latitude
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df3, x='pickup_longitude', y='pickup_latitude', alpha=0.1)
plt.title("Carte des points de départ des courses")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Render the matplotlib plot in Streamlit
st.pyplot()

# Plot a scatterplot of dropoff_longitude vs dropoff_latitude
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df3, x='dropoff_longitude', y='dropoff_latitude', alpha=0.1)
plt.title("Carte des points d'arrivée des courses")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Render the matplotlib plot in Streamlit
st.pyplot()

# Plot a scatterplot of trip_duration vs total_amount
plt.figure()
sns.scatterplot(data=df3, x='trip_duration', y='total_amount')
plt.title("Relation entre la durée des courses et le montant total payé")
plt.xlabel("Durée (minutes)")
plt.ylabel("Montant total")

# Render the matplotlib plot in Streamlit
st.pyplot()

import streamlit as st
import pydeck as pdk
import pandas as pd
import os

st.title("Visualisation de points pour les prises en charge et les déposes")

# Importer le dataset
@st.cache
def load_data():
    df = pd.read_csv("https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv")
    return df

df = load_data()

# Convertir les colonnes de dates en datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Créer une visualisation de points pour les prises en charge et les déposes
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1Ijoic2FtaWhzcyIsImEiOiJjbGdlN2ZrMmUwZ3N1M2ZtZnA2bTdtMXVsIn0.0vFWUtybsjQcElxgPoT-Mw"

pickup_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["pickup_longitude", "pickup_latitude"],
    get_radius=50,
    get_fill_color=[255, 140, 0],
    pickable=True,
    opacity=0.8
)

dropoff_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["dropoff_longitude", "dropoff_latitude"],
    get_radius=50,
    get_fill_color=[0, 128, 255],
    pickable=True,
    opacity=0.8
)

# Configurer la vue
view_state = pdk.ViewState(
    longitude=-73.98,
    latitude=40.75,
    zoom=11,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36
)

# Créer le rendu de la carte
r = pdk.Deck(
    layers=[pickup_layer, dropoff_layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>Color:</b> {color}", "style": {"color": "white"}},
    map_style="mapbox://styles/mapbox/light-v9"
)

# Afficher la carte
st.pydeck_chart(r)
