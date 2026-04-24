import streamlit as st
from transformers import pipeline


INDEX_NUMBER = "s28774"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-de"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


st.set_page_config(
    page_title="English to German Translator",
    page_icon="DE",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_translator():
    return pipeline("translation", model=TRANSLATION_MODEL)



@st.cache_resource(show_spinner=False)
def load_sentiment_classifier():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


st.title("English to German Translator")
st.caption("Aplikacja Streamlit wykorzystująca modele Hugging Face.")

st.info(
    "Wpisz tekst w języku angielskim, wybierz działanie i uruchom model. "
    "Aplikacja może tłumaczyć tekst na język niemiecki albo sprawdzić wydźwięk emocjonalny tekstu."
)

option = st.selectbox(
    "Opcje",
    [
        "Tłumaczenie tekstu EN -> DE",
        "Wydźwięk emocjonalny tekstu ",
    ],
)

text = st.text_area("Wpisz tekst po angielsku", height=160)

if st.button("Uruchom", type="primary"):
    if not text.strip():
        st.warning("Wpisz tekst, zanim uruchomisz model.")
    elif option == "Tłumaczenie tekstu EN -> DE":
        try:
            with st.spinner("Ładuję model i tłumaczę tekst..."):
                translator = load_translator()
                result = translator(text, max_length=512)

            translated_text = result[0]["translation_text"]
            st.success("Tłumaczenie gotowe.")
            st.subheader("Wynik")
            st.write(translated_text)
        except Exception as exc:
            st.error(f"Nie udało się wykonać tłumaczenia: {exc}")
    else:
        try:
            with st.spinner("Analizuję wydźwięk emocjonalny tekstu..."):
                classifier = load_sentiment_classifier()
                result = classifier(text)

            st.success("Analiza gotowa.")
            st.subheader("Wynik")
            st.json(result)
        except Exception as exc:
            st.error(f"Nie udało się wykonać analizy: {exc}")

st.divider()
st.write(f"Numer indeksu: {INDEX_NUMBER}")
