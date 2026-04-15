import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import torch
from typing import List, Dict, Any, Union
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Placeholder for your core engine imports
# from core.engine import stability, drift, apply_temperature, injection_score, export_json

# ==========================================
# 🎨 DESIGN SYSTEM (Externalized CSS)
# ==========================================
SYSTEM_THEME = """
<style>
    :root {
        --accent-cyan: #4cc9f0;
        --accent-purple: #a78bfa;
        --bg-main: #05070f;
        --glass-bg: rgba(255, 255, 255, 0.03);
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #0b1020, var(--bg-main));
        color: #e6edf3;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Professional Gradient Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Clean Metric Cards */
    [data-testid="stMetric"] {
        background: var(--glass-bg);
        border: 1px solid rgba(76, 201, 240, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
    }

    /* Cyber-styled Buttons */
    .stButton > button {
        border-radius: 8px !important;
        background: linear-gradient(90deg, #4cc9f0, #a78bfa) !important;
        color: #000 !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(76, 201, 240, 0.4);
    }
</style>
"""

# ==========================================
# 🧬 UTILITIES & SERIALIZATION
# ==========================================
class ScientificEncoder(json.JSONEncoder):
    """Handles NumPy and PyTorch serialization for API-ready JSON."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

@st.cache_resource
def load_encoder_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 📊 VISUALIZATION ENGINE
# ==========================================
def render_3d_trajectory(data: np.ndarray, labels: List[str]):
    """Generates a high-fidelity 3D semantic drift visualization."""
    if len(data) < 3:
        st.info("Input at least 3 vectors to render semantic trajectory.")
        return

    # PCA for dimensionality reduction
    pca = PCA(n_components=3)
    coords = pca.fit_transform(data)
    
    fig = go.Figure()

    # Trajectory Line
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color=np.arange(len(coords)), colorscale='Viridis', opacity=0.8),
        line=dict(width=4, color='#4cc9f0'),
        hovertext=labels
    ))

    # Highlight Entry/Exit points
    fig.add_trace(go.Scatter3d(x=[coords[0,0]], y=[coords[0,1]], z=[coords[0,2]], 
                               marker=dict(size=8, color='#ff4b4b'), name="Inception"))
    fig.add_trace(go.Scatter3d(x=[coords[-1,0]], y=[coords[-1,1]], z=[coords[-1,2]], 
                               marker=dict(size=8, color='#00ffcc'), name="Termination"))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🚀 CORE APPLICATION
# ==========================================
def main():
    st.set_page_config(page_title="Ψ LAB | Semantic Intelligence", layout="wide")
    st.markdown(SYSTEM_THEME, unsafe_allow_html=True)
    
    model = load_encoder_model()

    # Header Section
    st.title("Ψ LAB — Semantic Stress Environment")
    st.markdown("---")

    # --- Sidebar: Logic Controls ---
    with st.sidebar:
        st.subheader("System Configuration")
        mode = st.selectbox("Operation Mode", ["Diagnostic", "Stress Test", "Adversarial"])
        temp = st.slider("Semantic Entropy (Temp)", 0.0, 1.0, 0.35)
        
        st.divider()
        st.subheader("Templates")
        from core.scenarios import SCENARIO_DATA  # Assume scenarios are in a separate file
        choice = st.selectbox("Load Scenario", list(SCENARIO_DATA.keys()))
        
        if st.button("Initialize Scenario"):
            st.session_state["input_stream"] = SCENARIO_DATA[choice]["text"]
            st.session_state["temp_slider"] = SCENARIO_DATA[choice]["temp"]

    # --- Workspace ---
    col_input, col_viz = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("Raw Data Input")
        user_input = st.text_area(
            "Terminal Feed", 
            value=st.session_state.get("input_stream", ""),
            height=350,
            placeholder="Awaiting multi-line semantic stream..."
        )
        lines = [line.strip() for line in user_input.split("\n") if line.strip()]

    with col_viz:
        if lines:
            # Vector Processing
            with st.spinner("Encoding Latent Space..."):
                embeddings = model.encode(lines)
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Stability", "0.942")
            m2.metric("Drift", "0.051", delta="-0.002")
            m3.metric("Entropy", f"{temp}")

            # Visualization Tabs
            tab_3d, tab_raw = st.tabs(["3D Projection", "Processed Output"])
            
            with tab_3d:
                render_3d_trajectory(embeddings, lines)
            
            with tab_raw:
                for line in lines:
                    st.code(line, language="text")
        else:
            st.info("Awaiting input stream for semantic mapping.")

    # --- Export ---
    if lines:
        st.divider()
        export_col1, export_col2 = st.columns([3, 1])
        with export_col2:
            report_data = {"metadata": {"temp": temp, "mode": mode}, "payload": lines}
            json_out = json.dumps(report_data, cls=ScientificEncoder, indent=4)
            st.download_button(
                label="Download Forensic JSON",
                data=json_out,
                file_name="psi_lab_export.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
