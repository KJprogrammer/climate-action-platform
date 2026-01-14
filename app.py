"""
Climate Action Platform - GOODHack24 Top 3 Winner
Real-time environmental insights using GPT-4, LangChain, and RAG
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Climate Action Platform",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background: #f0f7ff;
        padding: 1rem;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def load_climate_data():
    """Load climate knowledge base"""
    climate_facts = [
        """Global temperatures have risen by approximately 1.1°C since pre-industrial times. 
        The last decade (2011-2020) was the warmest on record. Carbon dioxide levels are at 
        their highest in 3 million years at 415 parts per million.""",
        
        """Renewable energy costs have dropped dramatically: solar by 89%, wind by 70% since 2010. 
        Clean energy investments reached $500 billion in 2020. Electric vehicle sales are 
        growing at 40% annually.""",
        
        """Deforestation accounts for 10% of global emissions. Reforestation and afforestation 
        can sequester 5-10 gigatons of CO2 annually. Nature-based solutions could provide 
        37% of cost-effective CO2 mitigation by 2030.""",
        
        """Individual actions matter: switching to plant-based diets can reduce personal carbon 
        footprint by up to 73%. Energy-efficient appliances save 30% on electricity. 
        Public transit reduces emissions by 45% compared to driving.""",
        
        """Climate change impacts include: sea level rise of 3.3mm/year, increased extreme 
        weather events (up 400% since 1980), ocean acidification threatening marine ecosystems, 
        and species extinction rates 1000x higher than natural.""",
    ]
    
    return climate_facts

def create_vector_store(climate_data, api_key):
    """Create FAISS vector store from climate data"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    documents = []
    for text in climate_data:
        docs = text_splitter.create_documents([text])
        documents.extend(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def create_qa_chain(vector_store, api_key):
    """Create RAG QA chain with GPT-4"""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        openai_api_key=api_key
    )
    
    prompt_template = """You are a climate action expert. Use the following context to provide 
    actionable insights and recommendations for addressing climate change.
    
    Context: {context}
    
    Question: {question}
    
    Provide a detailed, actionable response with specific steps or solutions:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def generate_climate_metrics():
    """Generate sample climate impact metrics"""
    metrics = {
        'CO2 Reduction Potential': '2.5 GT/year',
        'Renewable Energy Growth': '+40% YoY',
        'Reforestation Target': '1 Billion Trees',
        'Individual Impact': '-2.3 tons CO2/year'
    }
    return metrics

def create_impact_visualization():
    """Create climate impact visualization"""
    data = {
        'Action': ['Renewable Energy', 'Reforestation', 'Sustainable Transport', 
                   'Energy Efficiency', 'Diet Change'],
        'CO2 Reduction (GT/year)': [5.2, 3.1, 2.8, 4.5, 6.2],
        'Cost Effectiveness': [9, 8, 7, 9, 10]
    }
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Action', y='CO2 Reduction (GT/year)',
                 color='Cost Effectiveness',
                 title='Climate Action Impact Analysis',
                 color_continuous_scale='Greens')
    
    return fig

# Main App
st.markdown('<h1 class="main-header">🌍 Climate Action Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Top 3 Winner - Waterloo GOODHack24 | Powered by GPT-4 & RAG</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", 
                            help="Enter your OpenAI API key")
    
    st.markdown("---")
    st.header("📊 Quick Stats")
    st.metric("Global Temp Rise", "+1.1°C", "since 1880")
    st.metric("CO2 Levels", "415 ppm", "highest in 3M years")
    st.metric("Renewable Growth", "+40%", "annually")
    
    st.markdown("---")
    st.info("💡 **Tip:** Ask about specific climate actions, impacts, or solutions!")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Climate Insights", "📈 Impact Analysis", "💡 Take Action"])

with tab1:
    st.header("Ask Climate Questions - Powered by RAG")
    
    if api_key:
        if st.session_state.vector_store is None:
            with st.spinner("Loading climate knowledge base..."):
                climate_data = load_climate_data()
                st.session_state.vector_store = create_vector_store(climate_data, api_key)
                st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store, api_key)
                st.success("✅ Climate knowledge base loaded!")
        
        query = st.text_input("Ask a climate question:", 
                             placeholder="e.g., What are the most effective individual actions to combat climate change?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Get Insights", type="primary")
        
        if ask_button and query:
            with st.spinner("Analyzing climate data..."):
                result = st.session_state.qa_chain.invoke({"query": query})
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 💡 Climate Insight")
                st.write(result['result'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:200] + "...")
        
        st.markdown("### 🌟 Sample Questions")
        sample_questions = [
            "What are the most cost-effective climate solutions?",
            "How can individuals reduce their carbon footprint?",
            "What is the impact of renewable energy adoption?",
            "How does deforestation affect climate change?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{i}"):
                    result = st.session_state.qa_chain.invoke({"query": q})
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.write(result['result'])
                    st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the Climate Insights feature.")

with tab2:
    st.header("Climate Impact Analysis")
    
    metrics = generate_climate_metrics()
    cols = st.columns(4)
    for i, (metric, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h3>{value}</h3><p>{metric}</p></div>', 
                       unsafe_allow_html=True)
    
    st.plotly_chart(create_impact_visualization(), use_container_width=True)
    
    st.markdown("### 📊 Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🌱 Top Climate Solutions:**
        - Renewable Energy: 5.2 GT CO2/year reduction
        - Diet Change: 6.2 GT CO2/year reduction
        - Energy Efficiency: 4.5 GT CO2/year reduction
        """)
    
    with col2:
        st.markdown("""
        **💰 Cost-Effectiveness:**
        - Diet Change: 10/10 (highest ROI)
        - Renewable Energy: 9/10
        - Energy Efficiency: 9/10
        """)

with tab3:
    st.header("Take Action Today")
    
    st.markdown("### 🎯 Personalized Climate Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏠 At Home:**
        - ☑️ Switch to LED bulbs (save 450 lbs CO2/year)
        - ☑️ Use programmable thermostat (save 15% energy)
        - ☑️ Reduce meat consumption (save 1,600 lbs CO2/year)
        - ☑️ Install solar panels (save 3-4 tons CO2/year)
        """)
        
        st.markdown("""
        **🚗 Transportation:**
        - ☑️ Use public transit (reduce 4,800 lbs CO2/year)
        - ☑️ Bike/walk for short trips
        - ☑️ Consider electric vehicle
        - ☑️ Carpool when possible
        """)
    
    with col2:
        st.markdown("""
        **🌳 Community:**
        - ☑️ Plant trees in your neighborhood
        - ☑️ Support renewable energy policies
        - ☑️ Join local climate action groups
        - ☑️ Educate others about climate change
        """)
        
        st.markdown("""
        **💼 Advocate:**
        - ☑️ Vote for climate-conscious leaders
        - ☑️ Support green businesses
        - ☑️ Reduce, reuse, recycle
        - ☑️ Invest in sustainable companies
        """)
    
    st.markdown("### 🧮 Quick Carbon Footprint Estimate")
    
    miles_driven = st.slider("Miles driven per week", 0, 500, 150)
    meat_meals = st.slider("Meat meals per week", 0, 21, 10)
    flights_year = st.slider("Flights per year", 0, 20, 2)
    
    transport_co2 = miles_driven * 52 * 0.89
    diet_co2 = meat_meals * 52 * 6.6
    flight_co2 = flights_year * 1100
    total_co2 = (transport_co2 + diet_co2 + flight_co2) / 2000
    
    st.markdown(f"""
    <div class="metric-card">
    <h2>Your Estimated Annual Carbon Footprint</h2>
    <h1>{total_co2:.1f} tons CO2</h1>
    <p>US Average: 16 tons | Global Average: 4 tons</p>
    </div>
    """, unsafe_allow_html=True)
    
    if total_co2 > 16:
        st.warning("⚠️ Your footprint is above the US average. Small changes can make a big impact!")
    else:
        st.success("✅ Great! You're below the US average. Keep up the good work!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏆 Top 3 Winner - Waterloo GOODHack24 | Built with GPT-4, LangChain & RAG</p>
    <p>Created by Kanwar Jhattu | Making climate action accessible through AI</p>
</div>
""", unsafe_allow_html=True)
