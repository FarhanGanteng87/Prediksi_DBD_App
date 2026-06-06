import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS (TEMA PINK)
# ==========================================
st.set_page_config(page_title="Prediksi Lama Rawat Inap Pasien DBD", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF0F5 !important; }
    .stApp p, .stApp label, .stApp span { color: #333333 !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #C71585 !important; font-weight: bold; }
    
    /* Input Form CSS */
    div[data-baseweb="select"] > div, input {
        background-color: #FFFFFF !important; 
        color: #333333 !important;            
        -webkit-text-fill-color: #333333 !important;
        border-radius: 6px !important;
        border: 1px solid #FFB6C1 !important; 
    }
    div[data-baseweb="select"] span { color: #333333 !important; }
    
    ul[role="listbox"], ul[data-baseweb="menu"] { background-color: #FFFFFF !important; }
    ul[role="listbox"] li, ul[data-baseweb="menu"] li, li[role="option"] {
        color: #333333 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar CSS */
    [data-testid="stSidebar"] { background-color: #FFE4E1 !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #000000 !important; }
    div[role="radiogroup"] label p, div[role="radiogroup"] label div { color: #000000 !important; }
    div[role="radiogroup"] > label > div:first-child {
        background-color: #FFFFFF !important;
        border-color: #FF69B4 !important;
    }
    
    /* Button CSS */
    .stButton button {
        background-color: #FFB6C1 !important; 
        color: #000000 !important;            
        border: 1px solid #FF69B4 !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: 0.3s;                     
    }
    .stButton button:hover {
        background-color: #FF69B4 !important; 
        color: #FFFFFF !important;            
        border: 1px solid #FF69B4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. FUNGSI LOAD DATA & PREPROCESSING
# ==========================================
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_excel(filepath, engine='xlrd')
    except:
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
        except:
            df = pd.read_csv(filepath)
            
    df = df.ffill().bfill() # Mengisi data kosong agar total pasien tetap utuh
    cols_to_drop = ['rm', 'tgl_masuk', 'tgl_keluar']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df

@st.cache_data
def preprocess_data(df):
    df_encoded = df.copy()
    le_jk, le_jd = LabelEncoder(), LabelEncoder()
    
    df_encoded['jenis_kelamin'] = le_jk.fit_transform(df_encoded['jenis_kelamin'])
    df_encoded['jenis_demam'] = le_jd.fit_transform(df_encoded['jenis_demam'])
    df_encoded['target'] = df_encoded['lama_dirawat'].apply(lambda x: 0 if x >= 5 else 1)
    
    scaler = MinMaxScaler()
    fitur = ['jenis_kelamin', 'umur', 'jenis_demam', 'hemoglobin', 'hct', 'trombosit']
    df_normalisasi = df_encoded.copy()
    df_normalisasi[fitur] = scaler.fit_transform(df_encoded[fitur])
    
    return df_encoded, df_normalisasi, scaler, le_jk, le_jd, fitur

@st.cache_resource
def train_model(df_normalisasi, fitur):
    X = df_normalisasi[fitur]
    y = df_normalisasi['target']
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    knn = KNeighborsClassifier()
    svm = SVC(probability=True, random_state=42)
    ann = MLPClassifier(max_iter=1000, random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[('LR', lr), ('KNN', knn), ('SVM', svm), ('ANN', ann)],
        voting='hard'
    )
    voting_clf.fit(X_balanced, y_balanced)
    return voting_clf

# ==========================================
# 3. INISIALISASI DATA & MODEL
# ==========================================
dataset_path = "dataset/dengue_fever_los_dataset.csv.xls"  
if not os.path.exists(dataset_path):
    st.error(f"❌ File dataset tidak ditemukan di `{dataset_path}`.")
    st.stop()

df = load_data(dataset_path)
df_encoded, df_normalisasi, scaler, le_jk, le_jd, fitur = preprocess_data(df)
model = train_model(df_normalisasi, fitur)

# ==========================================
# 4. SIDEBAR & NAVIGASI
# ==========================================
st.sidebar.title("🩺 Prediksi Lama Rawat Inap Pasien DBD")
menu = st.sidebar.radio("", ["1. Data Asli", "2. Preprocessing Data", "3. EDA", "4. Prediksi Lama Rawat Inap"])

# ==========================================
# 5. KONTEN HALAMAN BERDASARKAN MENU
# ==========================================
if menu == "1. Data Asli":
    st.title(" Data Rekam Medis Pasien DBD")
    st.dataframe(df)
    st.success(f"**Total Data:** {df.shape[0]} Baris | {df.shape[1]} Kolom")

elif menu == "2. Preprocessing Data":
    st.title("Preprocessing & Transformasi")
    st.markdown("### 1. Label Encoding & Target")
    st.write("Fokus target pada Lama Rawat Inap (0: Lama/Kritis, 1: Cepat).")
    st.dataframe(df_encoded.head(20))
    st.markdown("### 2. Normalisasi Data (MinMaxScaler)")
    st.dataframe(df_normalisasi[fitur].head(20))

elif menu == "3. EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Analisis relasi antar fitur dan hubungannya dengan Lama Rawat Inap pasien.")
    
    df_eda = df.copy()
    df_eda['hari_bersih'] = pd.to_numeric(df_eda['lama_dirawat'], errors='coerce').round().astype(int)
    df_eda['target_biner'] = np.where(df_eda['hari_bersih'] >= 5, 0, 1)
    df_eda['target_label'] = df_eda['target_biner'].map({
        0: 'Lama (>=5 hari)',
        1: 'Cepat (<5 hari)'
    })
    
    # 1. Statistik Deskriptif
    st.markdown("### 1. Statistik Deskriptif (Overview Data)")
    st.dataframe(df.describe().T, use_container_width=True)
    
    # 2. Distribusi Kelas Target
    st.markdown("### 2. Distribusi Kelas Target")
    col_t1, col_t2 = st.columns(2)
    counts = df_eda['target_biner'].value_counts().sort_index()
    labels = ['Lama (>=5 Hari)', 'Cepat (<5 Hari)']
    
    with col_t1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        bars = ax1.bar(labels, counts, color=['#FF1493', '#FFB6C1'], edgecolor='#C71585')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', color='#333333', fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)
        
    with col_t2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#FF1493', '#FFB6C1'], wedgeprops={'edgecolor': '#C71585'})
        st.pyplot(fig2)
    
    # 3. Histogram & KDE
    st.markdown("### 3. Histogram & KDE (Sebaran Fitur Berdasarkan Target)")
    st.write("Melihat penumpukan nilai (distribusi) pasien pada setiap fitur numerik.")
    
    fitur_numerik = ['umur', 'hemoglobin', 'hct', 'trombosit']
    df_hist = df_encoded.copy()
    df_hist['Status Rawat Inap'] = df_hist['target'].apply(lambda x: 'Lama (>=5 Hari)' if x == 0 else 'Cepat (<5 Hari)')
    
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(fitur_numerik):
        sns.histplot(data=df_hist, x=col, hue='Status Rawat Inap', kde=True, 
                     palette=['#FF1493', '#FFB6C1'], ax=axes[i], 
                     alpha=0.6, bins=20, line_kws={'linewidth': 2})
        
        axes[i].set_title(f"Distribusi {col.capitalize()} vs Lama Rawat", color='#C71585', fontweight='bold')
        axes[i].set_xlabel(f"Nilai {col.capitalize()}")
        axes[i].set_ylabel("Jumlah Pasien")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
    plt.tight_layout(pad=3.0)
    st.pyplot(fig3)
    
    # 4. Tren Jumlah Pasien Per Hari
    st.markdown("### 4. Tren Jumlah Pasien Berdasarkan Lama Hari Dirawat")
    st.write("Distribusi pasien sebelum dikelompokkan menjadi target biner (>=5 hari vs <5 hari).")
    
    urutan_hari = sorted(df_eda['hari_bersih'].unique())
    fig4, ax4 = plt.subplots(figsize=(11, 5))
    sns.countplot(data=df_eda, x='hari_bersih', order=urutan_hari, palette='RdPu', ax=ax4)
    ax4.set_title("Tren Distribusi Jumlah Pasien DBD Berdasarkan Lama Hari Dirawat", fontsize=12, pad=15)
    ax4.set_xlabel("Lama Dirawat (Hari)", fontsize=10)
    ax4.set_ylabel("Jumlah Pasien (Orang)", fontsize=10)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    
    for p in ax4.patches:
        if p.get_height() > 0:
            ax4.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    st.pyplot(fig4)
    
    # 5. Sebaran Fitur Kategorikal Berdasarkan Target
    st.markdown("### 5. Sebaran Fitur Kategorikal Berdasarkan Target")
    st.write("Perbandingan jumlah pasien untuk setiap kategori jenis kelamin dan jenis demam.")
    
    fitur_kategorikal = ['jenis_kelamin', 'jenis_demam']
    for col in fitur_kategorikal:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_eda, x=col, hue='target_label', 
                      palette={'Lama (>=5 hari)': '#FF1493', 'Cepat (<5 hari)': '#FFB6C1'}, ax=ax5)
        ax5.set_title(f"Perbandingan Fitur {col.upper()} Berdasarkan Target Rawat Inap")
        ax5.set_xlabel(col.upper())
        ax5.set_ylabel("Jumlah Pasien")
        ax5.grid(axis='y', linestyle='--', alpha=0.5)
        ax5.legend(title='Target')
        st.pyplot(fig5)
    
    # 6. Korelasi Antar Fitur Numerik (Pearson)
    st.markdown("### 6. Korelasi Antar Fitur Numerik (Pearson)")
    st.write("Korelasi linier antara variabel numerik: umur, hemoglobin, hct, trombosit.")
    
    matriks_korelasi = df_eda[fitur_numerik].corr(method='pearson')
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    sns.heatmap(matriks_korelasi, annot=True, cmap='RdPu', fmt='.2f', linewidths=0.5, ax=ax6)
    ax6.set_title('Matriks Korelasi Pearson (Fitur Numerik)', fontsize=11, pad=15)
    st.pyplot(fig6)
    
    # 7. Uji Chi-Square: Fitur Kategorikal vs Target
    st.markdown("### 7. Uji Chi-Square: Hubungan Fitur Kategorikal dengan Target")
    st.write("Menguji apakah jenis kelamin dan jenis demam memiliki hubungan signifikan dengan lama rawat inap (biner).")
    
    X_categorical = df_eda[['jenis_kelamin', 'jenis_demam']].copy()
    y_binary = df_eda['target_biner']
    
    for col in X_categorical.columns:
        X_categorical[col] = X_categorical[col].astype(str).astype('category').cat.codes
    
    chi2_scores, p_values = chi2(X_categorical, y_binary)
    
    hasil_chi2 = pd.DataFrame({
        'Fitur': ['jenis_kelamin', 'jenis_demam'],
        'Chi-Square Score': chi2_scores,
        'P-Value': p_values,
    })
    st.dataframe(hasil_chi2, use_container_width=True)
    st.caption("Catatan Medis: P-Value < 0.05 menunjukkan adanya hubungan signifikan dengan target lama rawat inap.")

elif menu == "4. Prediksi Lama Rawat Inap":
    st.title("Form Prediksi Lama Rawat Inap")
    
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            in_jk = st.selectbox("Jenis Kelamin", df['jenis_kelamin'].unique())
            in_umur = st.number_input("Umur (Tahun)", 1, 100, 25)
            in_jd = st.selectbox("Jenis Demam", df['jenis_demam'].unique())
        with col2:
            in_hb = st.number_input("Hemoglobin (g/dL)", 0.0, 30.0, 13.0)
            in_hct = st.number_input("Hematokrit (%)", 0.0, 100.0, 35.0)
            in_trombo = st.number_input("Trombosit", 0, 1000000, 150000)
        submit = st.form_submit_button("Mulai Prediksi")
        
    if submit:
        jk_enc = le_jk.transform([in_jk])[0]
        jd_enc = le_jd.transform([in_jd])[0]
        input_data = pd.DataFrame([[jk_enc, in_umur, jd_enc, in_hb, in_hct, in_trombo]], columns=fitur)
        input_scaled = scaler.transform(input_data)
        
        st.markdown("---")
        st.subheader("📊 Analisis Suara dari Setiap Metode (Individual)")
        
        for name, clf in model.named_estimators_.items():
            pred = clf.predict(input_scaled)[0]
            label_text = "Lama (>=5 Hari)" if pred == 0 else "Cepat (<5 Hari)"
            st.write(f"• **{name}** memprediksi: {label_text}")
            
        final_pred = model.predict(input_scaled)[0]
        st.markdown("---")
        st.subheader("Keputusan Akhir (Majority Voting)")
        if final_pred == 0:
            st.error("🩺 Pasien diprediksi **Lama Dirawat (>= 5 Hari)**")
        else:
            st.success("Pasien diprediksi **Cepat Pulang (< 5 Hari)**")