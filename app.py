import streamlit as st
import pandas as pd

from gensim import models
from gensim.corpora import Dictionary


# ==================================================
# KONFIGURASI
# ==================================================

st.set_page_config(
    page_title="BPJS Critical Issue Detector",
    page_icon="🏥",
    layout="wide"
)


# ==================================================
# LOAD MODEL LDA
# ==================================================

@st.cache_resource
def load_model():

    lda_model = models.LdaModel.load(
        "deployment/lda_model.model"
    )

    dictionary = Dictionary.load(
        "deployment/dictionary.dict"
    )

    return lda_model, dictionary


@st.cache_data
def load_label():

    df = pd.read_csv(
        "deployment/label_topik.csv"
    )

    return df



lda_model, dictionary = load_model()

label_topik = load_label()



# ==================================================
# PREPROCESS SEDERHANA
# ==================================================

def preprocess(text):

    text = text.lower()

    # hapus karakter sederhana
    text = (
        text
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
    )

    tokens = text.split()

    return tokens



# ==================================================
# FUNGSI PREDIKSI TOPIK
# ==================================================

def predict_topic(text):

    tokens = preprocess(text)


    bow = dictionary.doc2bow(
        tokens
    )


    topic_distribution = (
        lda_model
        .get_document_topics(
            bow
        )
    )


    if len(topic_distribution)==0:

        return None


    dominant_topic = max(
        topic_distribution,
        key=lambda x:x[1]
    )


    topic_id = (
        dominant_topic[0]
        + 1
    )


    probability = (
        dominant_topic[1]
        *100
    )


    return topic_id, probability



# ==================================================
# HEADER
# ==================================================

st.title(
    "🏥 BPJS Critical Issue Detector"
)


st.write(
    """
    Sistem identifikasi isu kritis layanan BPJS Kesehatan
    menggunakan pemodelan topik Latent Dirichlet Allocation (LDA).
    """
)



# ==================================================
# MENU
# ==================================================

menu = st.radio(
    "Pilih Mode Analisis",
    [
        "Single Text",
        "Upload CSV"
    ]
)



# ==================================================
# SINGLE TEXT
# ==================================================

if menu == "Single Text":


    st.subheader(
        "Analisis Satu Cuitan"
    )


    text_input = st.text_area(
        "Masukkan teks keluhan:"
    )


    if st.button(
        "Analisis"
    ):


        if text_input.strip():


            result = predict_topic(
                text_input
            )


            if result:


                topic, prob = result


                info = label_topik[
                    label_topik["Topik"]
                    == topic
                ]


                st.success(
                    "Analisis berhasil"
                )


                col1,col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Topik",
                        f"Topik {topic}"
                    )


                with col2:

                    st.metric(
                        "Probabilitas",
                        f"{prob:.2f}%"
                    )


                st.subheader(
                    "Identifikasi Isu Kritis"
                )


                st.info(
                    info.iloc[0]["Isu Kritis"]
                )


                st.write(
                    "Kata Dominan:"
                )


                st.write(
                    info.iloc[0]["Kata Dominan"]
                )



            else:

                st.warning(
                    "Tidak ditemukan topik"
                )


        else:

            st.warning(
                "Masukkan teks terlebih dahulu"
            )



# ==================================================
# BATCH CSV
# ==================================================

else:


    st.subheader(
        "Analisis Banyak Data"
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file:


        df = pd.read_csv(
            uploaded_file
        )


        st.write(
            "Preview Data"
        )

        st.dataframe(
            df.head()
        )


        kolom = st.selectbox(
            "Pilih kolom teks",
            df.columns
        )



        if st.button(
            "Proses CSV"
        ):


            hasil = []


            for text in df[kolom]:


                result = predict_topic(
                    str(text)
                )


                if result:

                    topic, prob = result


                    info = label_topik[
                        label_topik["Topik"]
                        ==
                        topic
                    ]


                    hasil.append(
                        {
                        "Teks":text,
                        "Topik":topic,
                        "Probabilitas":
                            round(prob,2),
                        "Isu Kritis":
                            info.iloc[0]["Isu Kritis"]
                        }
                    )



            df_hasil = pd.DataFrame(
                hasil
            )


            st.subheader(
                "Hasil Identifikasi"
            )


            st.dataframe(
                df_hasil,
                use_container_width=True
            )


            csv = df_hasil.to_csv(
                index=False
            )


            st.download_button(
                "Download Hasil CSV",
                csv,
                "hasil_identifikasi_isu.csv",
                "text/csv"
            )
