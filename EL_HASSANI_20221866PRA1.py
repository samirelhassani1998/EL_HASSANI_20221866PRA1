import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

def main():
    st.title("Day of the Month Distribution - Uber - April 2014")

    # Télécharger le fichier CSV à partir de Google Drive
    url = "https://drive.google.com/uc?id=1qoK_zLLWWPjY3HaPN4WXH2q1OayJiQtz"
    gdown.download(url, "uber-raw-data-apr14.csv", quiet=False)

    df = pd.read_csv("uber-raw-data-apr14.csv", delimiter=',')
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    def get_dom(dt):
        return dt.day

    df['dom'] = df['Date/Time'].map(get_dom)

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    fig = sns.histplot(df['dom'], bins=30, kde=False).get_figure()
    plt.title('Day of the Month Distribution - Uber - April 2014')
    plt.xlabel('Day of the Month')
    plt.ylabel('Frequency')
    
    st.pyplot(fig)

    def get_weekday(dt):
        return dt.weekday()

    df['weekday'] = df['Date/Time'].map(get_weekday)

    def get_hour(dt):
        return dt.hour

    df['hour'] = df['Date/Time'].map(get_hour)

    st.write(df.head())
    st.write(df.describe())
    st.write(df.info())

if __name__ == "__main__":
    main()
