import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Prediksi Hasil Pertandingan Sepakbola", layout="wide")

# ============ LOAD MODEL ============
@st.cache_resource
def load_models():
    dt_model = joblib.load("model_decision_tree.pkl")
    knn_model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return dt_model, knn_model, scaler, le

dt_model, knn_model, scaler, le = load_models()

FEATURE_COLUMNS = [
    'elo_diff', 'home_elo', 'away_elo',
    'home_avg_scored', 'home_avg_conceded', 'home_winrate', 'home_form',
    'away_avg_scored', 'away_avg_conceded', 'away_winrate', 'away_form',
    'neutral'
]

# ============ SIDEBAR ============
st.sidebar.title("⚽ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])
model_choice = st.sidebar.selectbox("Pilih Algoritma", ["Decision Tree", "K-Nearest Neighbors (KNN)"])

st.title("⚽ Prediksi Hasil Pertandingan Sepakbola Berdasarkan Statistik Tim")
st.markdown("Aplikasi ini memprediksi hasil pertandingan (**Home Win / Draw / Away Win**) berdasarkan statistik tim seperti ELO rating, rata-rata gol, dan form terkini.")

# ============ MAIN LOGIC ============
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head(20))
    st.write(f"Jumlah baris: **{df.shape[0]}**, Jumlah kolom: **{df.shape[1]}**")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing_cols:
        st.error(f"Dataset tidak memiliki kolom fitur yang dibutuhkan: {missing_cols}")
        st.info(f"Kolom yang dibutuhkan: {FEATURE_COLUMNS}")
    else:
        # ============ VISUALISASI DATA ============
        st.subheader("📈 Visualisasi Data")
        col1, col2 = st.columns(2)

        with col1:
            if 'result' in df.columns:
                fig, ax = plt.subplots()
                df['result'].value_counts().plot(kind='bar', color=['#4CAF50', '#F44336', '#FFC107'], ax=ax)
                ax.set_title("Distribusi Hasil Pertandingan")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.histplot(df['elo_diff'], kde=True, ax=ax, color='steelblue')
            ax.set_title("Distribusi Selisih ELO Rating (Home - Away)")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[FEATURE_COLUMNS].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Korelasi Antar Fitur")
        st.pyplot(fig)

        # ============ PREDIKSI ============
        st.subheader("🔮 Hasil Prediksi")
        X = df[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)

        model = dt_model if model_choice == "Decision Tree" else knn_model
        predictions = model.predict(X_scaled)
        predictions_label = le.inverse_transform(predictions)

        df_result = df.copy()
        df_result['Prediksi'] = predictions_label
        st.dataframe(df_result[['home_elo', 'away_elo', 'elo_diff', 'Prediksi'] + 
                                (['result'] if 'result' in df.columns else [])].head(50))

        # Distribusi hasil prediksi
        fig, ax = plt.subplots()
        pd.Series(predictions_label).value_counts().plot(kind='bar', color=['#4CAF50', '#F44336', '#FFC107'], ax=ax)
        ax.set_title(f"Distribusi Hasil Prediksi ({model_choice})")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

        # ============ EVALUASI (jika ada label asli) ============
        if 'result' in df.columns:
            st.subheader("✅ Evaluasi Model")
            y_true = le.transform(df['result'])
            acc = accuracy_score(y_true, predictions)
            st.metric("Akurasi", f"{acc:.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.text("Classification Report:")
                report = classification_report(y_true, predictions, target_names=le.classes_, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            with col2:
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_true, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

        # Download hasil
        csv_result = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Prediksi (CSV)", csv_result, "hasil_prediksi.csv", "text/csv")

else:
    st.info("👈 Silakan upload dataset (.csv) di sidebar untuk memulai.")
    st.markdown("""
    **Format kolom yang dibutuhkan:**
    - `elo_diff`, `home_elo`, `away_elo`
    - `home_avg_scored`, `home_avg_conceded`, `home_winrate`, `home_form`
    - `away_avg_scored`, `away_avg_conceded`, `away_winrate`, `away_form`
    - `neutral`
    - `result` (opsional, untuk evaluasi — isi: Home Win / Draw / Away Win)
    """)