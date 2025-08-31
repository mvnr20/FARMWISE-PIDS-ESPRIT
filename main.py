# ==============================
# FarmWise - Smart Farming App
# ==============================

from __future__ import annotations
import os
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- RAG Core Libraries ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="FarmWise — Smart Farming Assistant",
    page_icon="🌾",
    layout="wide",
)

# ----------------------------
# Helper dataclasses
# ----------------------------
@dataclass
class EnergyInputs:
    current_soil_moisture: float
    crop_stage: str
    evapotranspiration_mm: float
    expected_rain_mm: float
    pump_flow_m3_per_h: float
    electricity_price_per_kwh: float
    pump_kw: float
    area_hectares: float
    target_moisture: float

# ----------------------------
# Sidebar - Upload avec clarification
# ----------------------------
st.sidebar.header("⚙️ Charger vos datasets")
st.sidebar.markdown("""
**Instructions pour l'upload :**
- **Yield Dataset (CSV)** : doit contenir `Area`, `Country`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`. 
- **Sensor Dataset (CSV)** : `date`, `air_temp_c`, `soil_moisture_%`, `evapotranspiration_mm`, `rain_mm`, `energy_kwh`. 
- **Market Dataset (CSV)** : `date`, `market_price_per_kg`.
""")

yield_file = st.sidebar.file_uploader("Upload Yield Dataset", type=["csv"])
sensor_file = st.sidebar.file_uploader("Upload Sensor Dataset", type=["csv"])
market_file = st.sidebar.file_uploader("Upload Market Dataset", type=["csv"])

@st.cache_data
def load_csv(file, required_cols: List[str], default_df=None):
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.sidebar.error(f"⚠️ Colonnes manquantes: {missing}")
        return df
    else:
        return default_df

# Default datasets
dates = pd.date_range(end=datetime.today(), periods=30)
sensor_df_default = pd.DataFrame({
    "date": dates,
    "air_temp_c": np.random.uniform(15,25,30),
    "soil_moisture_%": np.random.uniform(40,70,30),
    "evapotranspiration_mm": np.random.uniform(2,5,30),
    "rain_mm": np.random.uniform(0,10,30),
    "energy_kwh": np.random.uniform(40,80,30)
})
market_df_default = pd.DataFrame({"date": dates, "market_price_per_kg": np.random.uniform(1.5,3.0,30)})
df_yield = pd.DataFrame() # Initialisation vide pour éviter l'erreur

sensor_df = load_csv(sensor_file, ["date","air_temp_c","soil_moisture_%","evapotranspiration_mm","rain_mm","energy_kwh"], sensor_df_default)
market_df = load_csv(market_file, ["date","market_price_per_kg"], market_df_default)
df_yield = load_csv(yield_file, ["Area","Country","Item","Year","hg/ha_yield","average_rain_fall_mm_per_year","pesticides_tonnes","avg_temp"], df_yield)

# Parse dates
for df in [sensor_df, market_df]:
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')

# ----------------------------
# Energy Optimization logic
# ----------------------------
CROP_STAGE_COEFF = {"seedling":0.7,"vegetative":1.0,"flowering":1.2,"fruiting":1.1}

def irrigation_need_mm(evapotranspiration_mm: float, crop_stage: str) -> float:
    k = CROP_STAGE_COEFF.get(crop_stage,1.0)
    return max(evapotranspiration_mm * k,0)

def irrigation_recommendation(inputs: EnergyInputs) -> Tuple[float,float,float]:
    gap_mm = max((inputs.target_moisture - inputs.current_soil_moisture)*0.5,0)
    base_need = irrigation_need_mm(inputs.evapotranspiration_mm, inputs.crop_stage)
    net_need_mm = max(base_need + gap_mm - inputs.expected_rain_mm,0)
    volume_m3 = net_need_mm * inputs.area_hectares * 10
    hours = volume_m3 / max(inputs.pump_flow_m3_per_h, 0.0001)
    energy_kwh = inputs.pump_kw * hours
    cost = energy_kwh * inputs.electricity_price_per_kwh
    return net_need_mm,hours,cost

# ----------------------------
# Yield forecasting logic
# ----------------------------
TARGET = "hg/ha_yield"
FEATURES = ["average_rain_fall_mm_per_year","pesticides_tonnes","avg_temp"]

@st.cache_resource
def train_yield_model_new(df):
    if df.empty:
        return None, None, None
    df_clean = df.dropna(subset=FEATURES+[TARGET])
    if df_clean.empty:
        return None, None, None
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    metrics = {"r2": r2_score(y,preds),"mae": mean_absolute_error(y,preds)}
    return model, metrics, scaler

def predict_yield_new(model, df, scaler):
    if model is None or scaler is None:
        return [0]
    # Corrected line to avoid FutureWarning
    X = df[FEATURES].ffill().bfill().values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)

def suggest_sell_window(market_df, horizon_days=14):
    if market_df.empty or len(market_df) < 14:
        return None, None, None
    df = market_df.sort_values("date").copy()
    df["ma7"] = df["market_price_per_kg"].rolling(7,min_periods=1).mean()
    df["ma14"] = df["market_price_per_kg"].rolling(14,min_periods=1).mean()
    future = df.tail(horizon_days)
    best_idx = future["market_price_per_kg"].idxmax()
    best_date = pd.to_datetime(df.loc[best_idx,"date"])
    start = (best_date - pd.Timedelta(days=2)).to_pydatetime()
    end = (best_date + pd.Timedelta(days=2)).to_pydatetime()
    best_price = float(df.loc[best_idx,"market_price_per_kg"])
    return pd.Timestamp(start), pd.Timestamp(end), best_price

# ----------------------------
# Header
# ----------------------------
st.title("🌾 FarmWise — Smart Farming Assistant")
st.markdown("""
**Objectifs Business** :
1. Réduction coûts énergétiques via irrigation intelligente. 
2. Prévision rendement & conseil vente selon prix marché.
""")
st.divider()

# --- RAG Core Functions ---
PROMPT_TEMPLATE = """
You are an expert agriculture assistant who creates detailed, practical, and easy-to-understand farming reports for farmers.

Use only the information provided in the following context. Do not include any information not mentioned in the context:

Context:
{context}

Generate a detailed farming report based on the context.

The report should be structured as follows:

1. Title: A clear title summarizing the topic.
2. Overview: A short introduction summarizing the main points from the context.
3. Key Findings: Bullet points highlighting the most important information.
4. Recommendations: Step-by-step actionable guidance.
5. Additional Notes: Any extra information from the context that might help the farmer.
6. Conclusion: A concise summary of the report’s key takeaways.

Instructions:
- Use simple, practical language suitable for farmers.
- Keep each section focused, clear, and concise.
- Do not include information outside the context.
- Do not justify your answers or explain why.
- Avoid phrases like "according to the context" or "mentioned above".

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_vector_store():
    db_dir = "./Farma_db"
    if os.path.exists(db_dir) and os.listdir(db_dir):
        # Load existing database
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = Chroma(
            collection_name="Farma_database",
            embedding_function=embedding_model,
            persist_directory=db_dir
        )
        st.success("✅ Existing knowledge base loaded.")
    else:
        # Create a new database
        os.makedirs(db_dir, exist_ok=True)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = Chroma(
            collection_name="Farma_database",
            embedding_function=embedding_model,
            persist_directory=db_dir
        )
        st.info("ℹ️ New knowledge base created.")
    return db

def process_documents(uploaded_files, db, progress_bar):
    if not uploaded_files:
        return "⚠️ No files provided to process."
    
    docs_to_add = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        try:
            st.markdown(f"**Processing:** {uploaded_file.name}...")
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()

            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name="sentence-transformers/all-mpnet-base-v2",
                chunk_size=100,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(data)
            docs_to_add.extend(chunks)

            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.remove(temp_file_path)
    
    if docs_to_add:
        st.markdown(f"**Adding {len(docs_to_add)} chunks to the database...**")
        db.add_documents(docs_to_add)
        st.session_state.ingested_files.extend([f.name for f in uploaded_files])
        return "✅ Documents processed and added to database."
    else:
        return "❌ No new documents were processed."

def run_query(query, db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=st.session_state.api_key,
        temperature=1
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | output_parser
    )

    result = rag_chain.invoke(query)
    return result

# --- State Management ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Get or create the vector store
st.session_state.db = get_vector_store()

# ----------------------------
# Tabs
# ----------------------------
tab_dash, tab_energy, tab_yield, tab_other = st.tabs([
    "📊 Dashboard",
    "⚡ Energy Optimization",
    "📈 Yield & Market",
    "🧰 Extras"
])

# ----------------------------
# Dashboard Tab
# ----------------------------
with tab_dash:
    st.subheader("Aperçu capteurs & marché")
    if not sensor_df.empty:
        latest = sensor_df.sort_values("date").iloc[-1]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🌡 Température", f"{latest['air_temp_c']:.1f} °C")
        k2.metric("💧 Humidité sol", f"{latest['soil_moisture_%']:.1f} %")
        k3.metric("☔ Pluie", f"{latest['rain_mm']:.1f} mm")
        k4.metric("⚡ Énergie", f"{latest['energy_kwh']:.1f} kWh")
        fig = px.line(sensor_df, x="date", y=["air_temp_c","soil_moisture_%","evapotranspiration_mm","rain_mm"], 
                      labels={"value":"Valeur","variable":"Mesure"}, title="Historique capteurs")
        st.plotly_chart(fig,use_container_width=True)
    if not market_df.empty:
        figp = px.line(market_df, x="date", y="market_price_per_kg", title="Prix marché ($/kg)")
        st.plotly_chart(figp,use_container_width=True)

# ----------------------------
# Energy Optimization Tab
# ----------------------------
with tab_energy:
    st.subheader("Recommandation irrigation / coût énergétique")
    if sensor_df.empty:
        st.warning("Dataset capteurs requis")
    else:
        last = sensor_df.sort_values("date").iloc[-1]
        c1,c2,c3 = st.columns(3)
        with c1:
            current_soil = st.slider("Humidité sol actuelle (%)",0.0,100.0,float(last["soil_moisture_%"]))
            evap = st.number_input("Évapotranspiration (mm/jour)",value=float(last["evapotranspiration_mm"]),step=0.1)
            expected_rain = st.number_input("Pluie attendue (mm/24h)",value=float(last["rain_mm"]),step=0.1)
        with c2:
            crop_stage = st.selectbox("Stade culture", list(CROP_STAGE_COEFF.keys()), index=1)
            target_moist = st.slider("Cible humidité sol (%)",10.0,90.0,60.0)
            area_ha = st.number_input("Surface (hectares)",value=1.0,step=0.1)
        with c3:
            pump_flow = st.number_input("Débit pompe (m³/h)",value=20.0,step=1.0)
            pump_kw = st.number_input("Puissance pompe (kW)",value=5.0,step=0.5)
            elec_price = st.number_input("Prix électricité ($/kWh)",value=0.15,step=0.01)

        inputs = EnergyInputs(
            current_soil_moisture=current_soil,
            crop_stage=crop_stage,
            evapotranspiration_mm=evap,
            expected_rain_mm=expected_rain,
            pump_flow_m3_per_h=pump_flow,
            electricity_price_per_kwh=elec_price,
            pump_kw=pump_kw,
            area_hectares=area_ha,
            target_moisture=target_moist
        )
        need_mm,hours,cost = irrigation_recommendation(inputs)
        k1,k2,k3 = st.columns(3)
        k1.metric("Eau à appliquer (mm)", f"{need_mm:.1f}")
        k2.metric("Durée pompe (h)", f"{hours:.2f}")
        k3.metric("Coût estimé ($)", f"{cost:.2f}")

# ----------------------------
# Yield & Market Tab (Corrected and Completed)
# ----------------------------
with tab_yield:
    st.subheader("Prévision rendement & comparaison multi-cultures")
    if df_yield.empty:
        st.warning("Dataset Yield requis pour cette section.")
    else:
        crops_selected = st.multiselect("Choisir cultures à comparer", df_yield['Item'].unique(), default=["Maize", "Wheat"])
        if crops_selected:
            fig_yield = go.Figure()
            for crop in crops_selected:
                df_crop = df_yield[df_yield['Item']==crop]
                fig_yield.add_trace(go.Scatter(
                    x=df_crop['Year'],
                    y=df_crop['hg/ha_yield'],
                    mode='markers',
                    name=crop
                ))
            fig_yield.update_layout(
                title="Rendement annuel par culture (hg/ha)",
                xaxis_title="Année",
                yaxis_title="Rendement (hg/ha)"
            )
            st.plotly_chart(fig_yield, use_container_width=True)

    st.subheader("Prévision de rendement basée sur les conditions")
    if df_yield.empty:
        st.warning("Dataset Yield requis pour la prévision.")
    else:
        st.markdown("Entrez les conditions pour la prévision de rendement :")
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_rain = st.number_input("Pluie moyenne annuelle (mm)", value=df_yield['average_rain_fall_mm_per_year'].mean())
        with c2:
            pesticides = st.number_input("Pesticides (tonnes)", value=df_yield['pesticides_tonnes'].mean())
        with c3:
            avg_temp = st.number_input("Température moyenne (°C)", value=df_yield['avg_temp'].mean())
        
        model, _, scaler = train_yield_model_new(df_yield)
        if model and scaler:
            input_df = pd.DataFrame([[avg_rain, pesticides, avg_temp]], columns=FEATURES)
            predicted_yield = predict_yield_new(model, input_df, scaler)[0]
            st.info(f"Rendement prévu: **{predicted_yield:.2f} hg/ha**")
        else:
            st.warning("Impossible de prévoir le rendement. Vérifiez le dataset Yield.")

    st.subheader("Suggestion de fenêtre de vente")
    if market_df.empty:
        st.warning("Dataset marché requis pour cette section.")
    else:
        start, end, price = suggest_sell_window(market_df)
        if start and end and price:
            st.success(f"""
            La meilleure fenêtre de vente basée sur les prix des 14 derniers jours est **entre le {start.strftime('%d %b')} et le {end.strftime('%d %b')}.**
            Le prix le plus élevé prévu dans cette période est de **{price:.2f} $/kg.**
            """)
        else:
            st.info("Pas assez de données pour suggérer une fenêtre de vente.")

# ----------------------------
# Extras Tab with new RAG feature
# ----------------------------
with tab_other:
    st.subheader("Performances du Modèle & Informations")
    st.markdown("### Modèle de Prévision de Rendement")
    
    if df_yield.empty:
        st.warning("Le dataset Yield est manquant.")
    else:
        try:
            _, metrics, _ = train_yield_model_new(df_yield)
            if metrics:
                st.info(f"""
                Le modèle de régression linéaire a été entraîné pour prédire le rendement. 
                Ses performances sur les données d'entraînement sont :
                - **Coefficient de détermination (R²)** : {metrics['r2']:.4f}
                - **Erreur absolue moyenne (MAE)** : {metrics['mae']:.2f} hg/ha
                """)
            else:
                st.warning("Le modèle n'a pas pu être entraîné avec les données fournies.")
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle : {e}")

    st.divider()
    
    # New RAG Assistant Feature
    st.title("🪴 FarmQuery - Your Agricultural AI Assistant")
    st.markdown(
        """
        Welcome, farmer! 🧑‍🌾 Upload your agricultural PDFs and get instant, context-aware answers.

        This app uses **Retrieval-Augmented Generation (RAG)** to provide accurate information from your own documents.
        - **Embedding Model**: HuggingFace `all-mpmpnet-base-v2` 🧠
        - **Vector Store**: ChromaDB 🗄️
        - **LLM**: Groq's `llama3-70b-8192` 🚀
        """
    )
    st.divider()
    
    with st.expander("⚙️ Settings & File Upload", expanded=True):
        st.session_state.api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.api_key,
            type="password",
            help="Find your key here: https://console.groq.com/keys"
        )
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True
        )
        
        st.markdown(f"**Indexed Files:** {', '.join(st.session_state.ingested_files) or 'None'}")
        
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Index Documents", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.warning("Please upload at least one PDF file to index.")
                else:
                    progress_bar = st.progress(0)
                    with st.spinner("Indexing documents..."):
                        message = process_documents(uploaded_files, st.session_state.db, progress_bar)
                        st.session_state.ingested_files = [f.name for f in uploaded_files]
                    progress_bar.empty()
                    st.success(message)
        with col2:
            if st.button("Clear Knowledge Base", use_container_width=True):
                if os.path.exists("./Farma_db"):
                    shutil.rmtree("./Farma_db")
                    st.session_state.ingested_files = []
                    st.session_state.db = get_vector_store()
                    st.rerun()
                    st.info("Knowledge base cleared. Please upload and re-index your documents.")
                else:
                    st.warning("No knowledge base to clear.")

    st.divider()

    question_input = st.text_area(
        "❓ Ask Your Question",
        placeholder="e.g., What are the best irrigation methods for corn? 🌽",
        height=100
    )

    if st.button("Get Answer", type="primary"):
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your Groq API key.")
        elif not question_input:
            st.error("⚠️ Please enter a question.")
        else:
            with st.spinner("🧠 Thinking... please wait ⏳"):
                try:
                    answer = run_query(question_input, st.session_state.db)
                    st.markdown("---")
                    st.subheader("RAG Answer")
                    st.write(answer)
                    st.success("✅ Done! Here is your answer. ✨")
                except Exception as e:
                    st.error(f"An error occurred: {e}")