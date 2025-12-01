"""
系统概览页面
展示整体框架设计和各层架构详情
"""

import streamlit as st
import plotly.graph_objects as go
from config.settings import CUSTOM_CSS

st.set_page_config(page_title="System Overview", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("📊 System Overview")
    st.markdown("### Comprehensive Framework Design")
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Architecture", 
        "🔄 Data Flow", 
        "🎯 Key Components",
        "📋 Technical Stack"
    ])
    
    with tab1:
        show_architecture()
    
    with tab2:
        show_data_flow()
    
    with tab3:
        show_key_components()
    
    with tab4:
        show_technical_stack()

def show_architecture():
    """显示系统架构"""
    st.markdown("## Multi-Layer Architecture")
    
    # 使用plotly创建架构图
    fig = go.Figure()
    
    layers = [
        "6. Fairness Monitoring & Accountability",
        "5. Explainability & Human Oversight",
        "4. Hybrid Deep Learning Core",
        "3. Fairness-Oriented Pre-Processing",
        "2. Privacy-Preserving Learning",
        "1. Data Security & Governance"
    ]
    
    colors = ['#f44336', '#ff9800', '#4caf50', '#2196F3', '#9c27b0', '#607d8b']
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        fig.add_trace(go.Bar(
            y=[layer],
            x=[100],
            orientation='h',
            marker=dict(color=color),
            text=layer,
            textposition='inside',
            hoverinfo='text',
            hovertext=f"Layer {6-i}: {layer}",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Six-Layer Framework Architecture",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False),
        height=400,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 各层详细说明
    st.markdown("### Layer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔒 Layer 1: Data Security & Governance"):
            st.markdown("""
            **Purpose**: Protect sensitive DV data from unauthorized access
            
            **Key Mechanisms**:
            - De-identification & Pseudonymization
            - End-to-end encryption (AES-256)
            - Role-Based Access Control (RBAC)
            - Data minimization principles
            - Audit logging
            
            **Compliance**: GDPR, CIA/IA principles
            """)
        
        with st.expander("🛡️ Layer 2: Privacy-Preserving Learning"):
            st.markdown("""
            **Purpose**: Train models without exposing individual records
            
            **Key Mechanisms**:
            - Differential Privacy (ε = 1.0)
            - Federated Learning
            - Secure Multi-Party Computation (SMPC)
            - Privacy budget management
            
            **Benefit**: Inter-agency collaboration without data centralization
            """)
        
        with st.expander("⚖️ Layer 3: Fairness-Oriented Pre-Processing"):
            st.markdown("""
            **Purpose**: Address representation and measurement biases
            
            **Key Mechanisms**:
            - Reweighing underrepresented groups
            - SMOTE oversampling
            - Targeted imputation for high-missingness groups
            - Removal of discriminatory proxy variables
            
            **Target**: Migrant, rural, and economically disadvantaged women
            """)
    
    with col2:
        with st.expander("🤖 Layer 4: Hybrid Deep Learning Core"):
            st.markdown("""
            **Purpose**: Predict DV risk from temporal sequences
            
            **Architecture**:
            - BiLSTM: Short-term temporal dependencies
            - Transformer: Long-range relational patterns
            - Attention mechanism for interpretability
            
            **Output**:
            - Continuous risk score (0-1)
            - Discrete risk level (Low/Medium/High)
            """)
        
        with st.expander("👁️ Layer 5: Explainability & Human Oversight"):
            st.markdown("""
            **Purpose**: Maintain accountability and prevent automated harm
            
            **Key Mechanisms**:
            - SHAP-based local explanations
            - Attention weight visualization
            - Mandatory human review for high-risk cases
            - Override capability for trained professionals
            
            **Principle**: Human-in-the-loop decision making
            """)
        
        with st.expander("📊 Layer 6: Fairness Monitoring & Accountability"):
            st.markdown("""
            **Purpose**: Prevent long-term drift and disparate impacts
            
            **Key Mechanisms**:
            - Continuous FPR/FNR monitoring across subgroups
            - Automated retraining triggers
            - Immutable audit logs
            - Group-specific threshold calibration
            
            **Compliance**: EU AI Act requirements for high-risk AI
            """)

def show_data_flow():
    """显示数据流程"""
    st.markdown("## Data Pipeline")
    
    # 使用Sankey图展示数据流
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Raw Data Sources", 
                   "Data Governance", 
                   "Privacy Layer",
                   "Pre-processing",
                   "Model Training",
                   "Prediction",
                   "Explainability",
                   "Human Review",
                   "Final Decision"],
            color=["#607d8b", "#9c27b0", "#2196F3", "#4caf50", 
                   "#ff9800", "#f44336", "#e91e63", "#795548", "#000000"]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6, 7],
            target=[1, 2, 3, 4, 5, 6, 7, 8],
            value=[100, 95, 90, 85, 80, 75, 70, 65]
        )
    )])
    
    fig.update_layout(
        title="End-to-End Data Flow",
        height=400,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📌 Key Points</h4>
    <ul>
        <li><strong>Data Sources</strong>: Police reports, medical records, household info, residential data</li>
        <li><strong>Governance</strong>: Anonymization reduces identifiable information by ~5%</li>
        <li><strong>Privacy</strong>: DP and FL protect individual records during training</li>
        <li><strong>Pre-processing</strong>: Bias mitigation ensures fair representation</li>
        <li><strong>Prediction</strong>: Hybrid model generates risk scores</li>
        <li><strong>Explainability</strong>: SHAP provides case-level justifications</li>
        <li><strong>Human Review</strong>: Mandatory for borderline/high-risk cases</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_key_components():
    """显示关键组件"""
    st.markdown("## Core Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>🔐 Privacy Technologies</h3>
        <ul>
            <li><strong>Differential Privacy</strong><br/>ε = 1.0, noise injection</li>
            <li><strong>Federated Learning</strong><br/>Decentralized training</li>
            <li><strong>Homomorphic Encryption</strong><br/>Computation on encrypted data</li>
            <li><strong>Secure Aggregation</strong><br/>SMPC protocols</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>🎯 Fairness Techniques</h3>
        <ul>
            <li><strong>Pre-processing</strong><br/>Reweighing, SMOTE</li>
            <li><strong>In-processing</strong><br/>Fair loss optimization</li>
            <li><strong>Post-processing</strong><br/>Threshold calibration</li>
            <li><strong>Monitoring</strong><br/>Continuous FPR/FNR tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>💡 Explainability Tools</h3>
        <ul>
            <li><strong>SHAP Values</strong><br/>Feature importance</li>
            <li><strong>Attention Weights</strong><br/>Temporal patterns</li>
            <li><strong>Counterfactuals</strong><br/>What-if scenarios</li>
            <li><strong>Rule Extraction</strong><br/>Interpretable rules</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_technical_stack():
    """显示技术栈"""
    st.markdown("## Technical Implementation")
    
    tech_categories = {
        "Deep Learning": ["PyTorch", "TensorFlow", "Transformers Library"],
        "Privacy": ["Opacus (DP)", "PySyft (FL)", "TenSEAL (HE)"],
        "Fairness": ["AIF360", "Fairlearn", "Themis-ML"],
        "Explainability": ["SHAP", "LIME", "Captum"],
        "Data Processing": ["Pandas", "NumPy", "Scikit-learn"],
        "Visualization": ["Plotly", "Matplotlib", "Seaborn"],
        "Security": ["Cryptography", "HashiCorp Vault", "AWS KMS"],
        "Deployment": ["Streamlit", "Docker", "Kubernetes"]
    }
    
    col1, col2 = st.columns(2)
    
    categories = list(tech_categories.items())
    mid = len(categories) // 2
    
    with col1:
        for category, tools in categories[:mid]:
            st.markdown(f"**{category}**")
            for tool in tools:
                st.markdown(f"- {tool}")
            st.markdown("")
    
    with col2:
        for category, tools in categories[mid:]:
            st.markdown(f"**{category}**")
            for tool in tools:
                st.markdown(f"- {tool}")
            st.markdown("")
    
    # 模型架构细节
    st.markdown("---")
    st.markdown("### Model Architecture Details")
    
    code_col1, code_col2 = st.columns(2)
    
    with code_col1:
        st.code("""
# BiLSTM Component
class BiLSTMEncoder(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        output, (h_n, c_n) = self.lstm(x)
        return output
        """, language="python")
    
    with code_col2:
        st.code("""
# Transformer Component
class TransformerEncoder(nn.Module):
    def __init__(self):
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dropout=0.3
            ),
            num_layers=6
        )
    
    def forward(self, x):
        # x: (seq_len, batch, features)
        output = self.transformer(x)
        return output
        """, language="python")

if __name__ == "__main__":
    main()
