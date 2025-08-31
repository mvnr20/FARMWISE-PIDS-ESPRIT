# 🌾 FarmWise - Smart Farming Assistant

FarmWise is a comprehensive web application for optimizing farming operations through data-driven insights.

---

## 🌟 Business Objectives

- **Cost Reduction:** Optimize energy consumption by providing intelligent, data-based irrigation recommendations.  
- **Revenue Maximization:** Forecast crop yield and suggest the best market window for selling products to achieve the highest prices.  

---

## 🚀 Features

### For Farmers & Agribusiness Professionals (Non-Technical)

#### 📊 Dashboard
Get a real-time overview of key sensor data, including temperature, soil moisture, and rainfall. Visualize historical trends of these metrics and market prices on a clean, interactive dashboard to monitor your farm's health at a glance.

#### ⚡ Energy Optimization
Input your farm's specifics (crop type, area, pump details) and receive a tailored irrigation recommendation. The app calculates the exact amount of water needed, estimated pump duration, and total energy cost—helping you save money and conserve resources. The intuitive interface with sliders and input fields makes it easy to experiment with different scenarios.

#### 📈 Yield & Market Analysis
Upload your historical yield data to train a predictive model. The app forecasts future crop yield based on conditions like average rainfall, pesticide use, and temperature. It also analyzes market data to identify the optimal 4-day window to sell your products for the best price, maximizing revenue.

#### 🧰 AI Assistant (FarmQuery)
A powerful AI chatbot providing instant, context-aware answers to your agricultural questions. Upload your PDF documents (farming manuals, research papers, etc.), and the AI transforms them into a searchable knowledge base for accurate and practical advice.

---

## 🔧 How to Use the App

### Step 1: Upload Your Datasets
On the sidebar, you'll find three upload buttons. To enable all features, please upload the following CSV files:  
- **Yield Dataset:** For yield forecasting.  
- **Sensor Dataset:** For the dashboard and energy optimization.  
- **Market Dataset:** For market price analysis.


### Step 2: Navigate the Tabs
- **Dashboard:** View sensor and market price data at a glance.  
- **Energy Optimization:** Adjust sliders and inputs to get irrigation recommendations and estimated costs.  
- **Yield & Market:** Compare crops, forecast yields, and get a suggested selling window.

### Extras (AI Assistant)
1. Enter your Groq API Key.  
2. Upload one or more agricultural PDF documents.  
3. Click **Process Documents** to build a database from your files.  
4. Ask a question in the text box and click **Get Answer** for a response based on your documents.

---

## 💾 Downloading the Files
To download this file, copy the content and paste it into a text editor. Save it with a `.md` extension (e.g., `README.md`). Repeat the process for other project files, such as `farmwise_app.py` and `requirements.txt`.

---

## 💻 For Developers (Technical Details)

### Technology Stack
- **Frontend/Web Framework:** Streamlit  
- **Data Handling:** Pandas, NumPy  
- **Data Visualization:** Plotly  
- **Machine Learning:** Scikit-learn (Linear Regression)

### AI / RAG
- **LLM:** Groq's llama3-70b-8192 model  
- **Embedding Model:** HuggingFace `all-mpnet-base-v2`  
- **Vector Database:** ChromaDB  
- **Orchestration:** LangChain

### Local Setup

1. **Prerequisites:** Python 3.10+ installed.  
2. **Clone the Repository:**  
   ```
   git clone https://github.com/mvnr20/FARMWISE-PIDS.git
   ```
3. **Create Virtual Environment & Activate It**
```
python -m venv venv
venv\Scripts\activate 
```
4. **Install Requirements**
```
pip install -r requirements.txt
```
5. **Run the Streamlit App**
```
streamlit run main.py

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
How can effective post-harvest storage technologies benefit smallholder farmers?
```