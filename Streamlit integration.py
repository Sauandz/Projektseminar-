#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

# Speicherort der Dateien
base_path = '/Users/sauanmahmud/Desktop/Daten/'
data_file = os.path.join(base_path, 'Realisierte_Erzeugung_201701010000_202301010000_Tag.csv')
data_file_2023 = os.path.join(base_path, 'Realisierte_Erzeugung_202301010000_202401010000_Tag.csv')
consumption_file = os.path.join(base_path, 'Realisierter_Stromverbrauch_201701010000_202301010000_Tag.csv')

# Funktion zur Bereinigung und Umwandlung von numerischen Werten
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = (df[col]
                   .astype(str)
                   .str.replace('.', '', regex=False)
                   .str.replace(',', '.', regex=False)
                   .astype(float))

# Funktion zum Bereinigen und Verarbeiten der Daten
def process_and_visualize(file_path_2017_2023, file_path_2023, consumption_file):
    if os.path.exists(file_path_2017_2023) and os.path.exists(file_path_2023) and os.path.exists(consumption_file):
        # Daten laden
        data_2017_2023 = pd.read_csv(file_path_2017_2023, sep=';', decimal=',')
        data_2023 = pd.read_csv(file_path_2023, sep=';', decimal=',')
        consumption_data = pd.read_csv(consumption_file, sep=';', decimal=',')

        # Datum konvertieren
        data_2017_2023['Datum'] = pd.to_datetime(data_2017_2023['Datum von'], format='%d.%m.%Y')
        data_2023['Datum'] = pd.to_datetime(data_2023['Datum von'], format='%d.%m.%Y')
        consumption_data['Datum'] = pd.to_datetime(consumption_data['Datum von'], format='%d.%m.%Y')

        renewable_cols = [
            "Biomasse [MWh] Berechnete Auflösungen",
            "Wasserkraft [MWh] Berechnete Auflösungen",
            "Wind Offshore [MWh] Berechnete Auflösungen",
            "Wind Onshore [MWh] Berechnete Auflösungen",
            "Photovoltaik [MWh] Berechnete Auflösungen",
            "Sonstige Erneuerbare [MWh] Berechnete Auflösungen",
        ]

        # Datenbereinigung
        clean_numeric_columns(data_2017_2023, renewable_cols)
        clean_numeric_columns(data_2023, renewable_cols)
        clean_numeric_columns(consumption_data, ["Gesamt (Netzlast) [MWh] Berechnete Auflösungen"])

        # Summieren der erneuerbaren Energien
        data_2017_2023['Total Renewables [MWh]'] = data_2017_2023[renewable_cols].sum(axis=1)
        data_2023['Total Renewables [MWh]'] = data_2023[renewable_cols].sum(axis=1)

        # Kombinieren der Daten
        combined_generation = pd.concat([data_2017_2023[['Datum', 'Total Renewables [MWh]']], data_2023[['Datum', 'Total Renewables [MWh]']]])
        combined_data = pd.merge(combined_generation, consumption_data[['Datum', 'Gesamt (Netzlast) [MWh] Berechnete Auflösungen']], on='Datum', how='inner')

        # Umbenennen
        combined_data.rename(columns={"Gesamt (Netzlast) [MWh] Berechnete Auflösungen": "Total Consumption [MWh]"}, inplace=True)

        # Regressionsanalyse vorbereiten
        combined_data['Year'] = combined_data['Datum'].dt.year
        combined_data['Month'] = combined_data['Datum'].dt.month
        combined_data['Day'] = combined_data['Datum'].dt.day

        X = combined_data[['Year', 'Month', 'Day', 'Total Consumption [MWh]']]
        y = combined_data['Total Renewables [MWh]']

        # Feature Selection
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        scores = selector.scores_

        # Wichtigkeit der Features anzeigen
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Score': scores}).sort_values(by='Score', ascending=False)

        # Cross-Validation
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        model_results = {}
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            model_results[name] = {
                'Mean R^2': np.mean(cv_scores),
                'Std': np.std(cv_scores)
            }

        # Visualisierung
        st.title("Prognose der Erneuerbaren Energien")
        st.write("### Wichtigkeit der Features")
        st.dataframe(feature_importance)

        st.write("### Modellbewertungen")
        st.dataframe(pd.DataFrame(model_results).T)

        for feature in ['Year', 'Month', 'Day']:
            grouped = combined_data.groupby(feature).agg({'Total Renewables [MWh]': 'sum', 'Total Consumption [MWh]': 'sum'}).reset_index()
            st.write(f"### {feature}-weise Produktion und Verbrauch")
            st.bar_chart(grouped.set_index(feature))

        # Berechnung des Gesamtverbrauchs und der Gesamterzeugung
        total_consumption = combined_data['Total Consumption [MWh]'].sum()
        total_renewables = combined_data['Total Renewables [MWh]'].sum()
        st.write(f"## Gesamter Stromverbrauch: {total_consumption:.2f} MWh")
        st.write(f"## Gesamte erneuerbare Erzeugung: {total_renewables:.2f} MWh")

process_and_visualize(data_file, data_file_2023, consumption_file)



# In[2]:


import xgboost as xgb

# XGBoost Modell hinzufügen
models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Cross-Validation für das XGBoost-Modell durchführen
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name} - Mean R^2: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")


# In[ ]:




