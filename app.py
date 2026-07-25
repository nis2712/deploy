import streamlit as st
import pandas as pd
import plotly.express as px


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="BPJS Critical Issue Insight Dashboard",
    page_icon="🏥",
    layout="wide"
)


# =====================================================
# JUDUL DASHBOARD
# =====================================================

st.title("🏥 BPJS Critical Issue Insight Dashboard")

st.markdown(
    """
    Dashboard ini menampilkan hasil identifikasi isu kritis
    layanan BPJS Kesehatan berdasarkan pemodelan topik
    Latent Dirichlet Allocation (LDA) pada data cuitan
    bersentimen negatif di media sosial X.
    """
)


# =====================================================
# LOAD DATA HASIL LDA
# =====================================================

@st.cache_data
def load_data():

    df_topik = pd.read_csv(
        "deployment/hasil_topik.csv"
    )

    df_keyword = pd.read_csv(
        "deployment/kata_dominan.csv"
    )

    df_tweet = pd.read_csv(
        "deployment/contoh_tweet.csv"
    )

    return df_topik, df_keyword, df_tweet


df_topik, df_keyword, df_tweet = load_data()



# =====================================================
# INFORMASI DATASET
# =====================================================

st.subheader("📊 Informasi Dataset")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Jumlah Data Analisis LDA",
        len(df_topik)
    )


with col2:
    st.metric(
        "Jumlah Topik Optimal",
        df_topik["Topik"].nunique()
    )


with col3:
    st.metric(
        "Metode",
        "LDA"
    )



# =====================================================
# DISTRIBUSI TOPIK
# =====================================================

st.divider()

st.subheader(
    "📌 Distribusi Isu Kritis"
)


jumlah_topik = (
    df_topik["Topik"]
    .value_counts()
    .reset_index()
)


jumlah_topik.columns = [
    "Topik",
    "Jumlah Cuitan"
]


jumlah_topik = jumlah_topik.sort_values(
    "Topik"
)


fig = px.bar(
    jumlah_topik,
    x="Topik",
    y="Jumlah Cuitan",
    text="Jumlah Cuitan",
    title="Distribusi Jumlah Cuitan Setiap Topik"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# =====================================================
# IDENTIFIKASI ISU KRITIS
# =====================================================

st.divider()

st.subheader(
    "🔎 Identifikasi Isu Kritis"
)


pilihan_topik = st.selectbox(
    "Pilih Topik",
    sorted(
        df_keyword["Topik"].unique()
    )
)



# mengambil kata dominan

detail_topik = df_keyword[
    df_keyword["Topik"]
    ==
    pilihan_topik
]


if len(detail_topik) > 0:

    st.markdown(
        "### Kata Dominan"
    )

    st.info(
        detail_topik.iloc[0]
        ["Kata Dominan"]
    )



# =====================================================
# CONTOH CUITAN
# =====================================================

st.markdown(
    "### Contoh Cuitan"
)


# mengambil nomor topik

nomor_topik = int(
    pilihan_topik
    .replace(
        "Topik ",
        ""
    )
)



contoh = df_tweet[
    df_tweet["Topik"]
    ==
    nomor_topik
]



if len(contoh) > 0:

    for tweet in contoh.iloc[:, 1]:

        st.write(
            "•",
            tweet
        )

else:

    st.warning(
        "Contoh cuitan tidak ditemukan."
    )



# =====================================================
# TABEL SEMUA TOPIK
# =====================================================

st.divider()

st.subheader(
    "📋 Ringkasan Seluruh Topik"
)


st.dataframe(
    df_keyword,
    use_container_width=True
)



# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    """
    Model:
    IndoBERTweet hasil continued fine-tuning + 
    Latent Dirichlet Allocation (LDA)

    Sumber data:
    Media sosial X
    """
)
