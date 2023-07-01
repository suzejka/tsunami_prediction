import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json

def read_data():
    raw_data = pd.read_csv("raw_tsunami_data.csv")
    preprocessed_data = pd.read_csv("preprocessed_tsunami_data.csv")
    with open('results_of_models.json', 'rb') as f:
        results_of_models = json.load(f)

    raw_data = raw_data.drop(raw_data.columns[0], axis=1)
    preprocessed_data = preprocessed_data.drop(preprocessed_data.columns[0], axis=1)

    return raw_data, preprocessed_data, results_of_models

def transform_labeled_data(data):
    with open('label_encoder_for_places.pickle', 'rb') as f:
        label_encoder_for_places = pickle.load(f)
    data["place_country"] = label_encoder_for_places.inverse_transform(data["place_country"])
    return data

def main():
    st.title("Przewidywanie tsunami podczas trzęsienia ziemi")
    st.markdown("Dane pochodzą ze strony https://earthquake.usgs.gov/data/comcat/index.php")
    raw_data, preprocessed_data, results_of_models = read_data()

    col1, col2 = st.columns(2)
    with col1:
        st.header("Dane surowe")
        st.dataframe(raw_data)
        if st.button("Wyświetl rozmiar danych surowych"):
            st.text(f"Rozmiar danych surowych: {str(raw_data.shape)}")
    with col2:
        st.header("Dane przetworzone")
        st.dataframe(preprocessed_data)
        if st.button("Wyświetl rozmiar danych przetworzonych"):
            st.text(f"Rozmiar danych przetworzonych: {str(preprocessed_data.shape)}")

    st.header("Miejsca z największą liczbą trzęsień ziemi")
    preprocessed_data_with_fixed_labels = transform_labeled_data(preprocessed_data)
    st.bar_chart(preprocessed_data_with_fixed_labels["place_country"].value_counts()[preprocessed_data_with_fixed_labels["place_country"].value_counts() > 450])
    st.text("Najwięcej trzęsień ziemi jest w Indonezji oraz na Sandwich Islands - to te obszary są najbardziej zagrożone, więc warto tam zainstalować systemy ostrzegania przed trzęsieniami ziemi")

    st.header("Miejsca z najmniejszą liczbą trzęsień ziemi")
    st.bar_chart(preprocessed_data_with_fixed_labels["place_country"].value_counts()[preprocessed_data_with_fixed_labels["place_country"].value_counts() < 10])
    st.text("Widać, że najmniej trzęsień ziemi jest na morzu egejskim oraz w USA")

    st.header("Histogram magnitud")
    fig, ax = plt.subplots()
    ax.hist(preprocessed_data['mag'], bins=20)
    st.pyplot(fig)

    st.text("Histogram magnitud pokazuje, że najwięcej trzęsień ziemi ma magnitudę 4.5. Trzęsienia ziemi o magnitudzie powyżej 7 występują bardzo rzadko.")

    geo_data = preprocessed_data[preprocessed_data["mag"] > 7][["lat", "lon"]]
    st.header("Mapa z miejscami trzęsień ziemi")
    st.map(geo_data, zoom=3)

    counts = []
    model_names = []

    for key in results_of_models:
        single_quotes = key.count("'")
        if single_quotes > 0:
            single_quotes = single_quotes / 2
        counts.append(int(single_quotes))
        model_names.append(key.split("without")[0])

    st.header("Wyniki modeli uczenia maszynowego")

    results_of_models = pd.DataFrame.from_dict(results_of_models, orient='index', columns=["Dokładność"])
    st.dataframe(results_of_models.sort_values(by="Dokładność", ascending=False))

if __name__ == "__main__":
    main()




    