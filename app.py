"""
SDG 5-Oriented Domestic Violence Risk Assessment System
主应用入口文件
"""

import streamlit as st
from config.settings import PAGE_CONFIG, CUSTOM_CSS

# 页面配置
st.set_page_config(**PAGE_CONFIG)

# 自定义CSS样式
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    """主页面"""
    
    # 页面标题
    st.title("🛡️ DV Risk Assessment System")
    st.markdown("### Advancing Gender Equality through AI (SDG 5)")
    
    # 欢迎信息
    st.markdown("""
    <div class="info-box">
    <h4>Welcome to the Domestic Violence Risk Assessment System</h4>
    <p>This is an ethical, privacy-preserving, and bias-mitigated AI framework designed to 
    support safe and equitable interventions for domestic violence prevention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 系统特性展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>🔒 Privacy-Preserving</h3>
        <ul>
            <li>Differential Privacy</li>
            <li>Federated Learning</li>
            <li>End-to-End Encryption</li>
            <li>Data Minimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>⚖️ Ethically Aligned</h3>
        <ul>
            <li>Non-maleficence</li>
            <li>Justice & Fairness</li>
            <li>Human Oversight</li>
            <li>Accountability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>🎯 Bias-Mitigated</h3>
        <ul>
            <li>Pre-processing Mitigation</li>
            <li>Fair Loss Optimization</li>
            <li>Group-specific Calibration</li>
            <li>Continuous Monitoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 系统架构概览
    st.markdown("## 🏗️ System Architecture")
    
    architecture_layers = [
        ("1️⃣", "Data Security & Governance Layer", 
         "De-identification, encryption, RBAC, data minimization"),
        ("2️⃣", "Privacy-Preserving Learning Layer", 
         "Differential privacy, federated learning, SMPC"),
        ("3️⃣", "Fairness-Oriented Pre-Processing Layer", 
         "Reweighing, SMOTE oversampling, targeted imputation"),
        ("4️⃣", "Hybrid Deep Learning Core", 
         "BiLSTM + Transformer architecture for temporal risk prediction"),
        ("5️⃣", "Explainability & Human Oversight Layer", 
         "SHAP analysis, attention visualization, human-in-the-loop"),
        ("6️⃣", "Fairness Monitoring & Accountability Layer", 
         "Continuous FPR/FNR monitoring, audit logs, retraining triggers")
    ]
    
    for icon, layer, desc in architecture_layers:
        with st.expander(f"{icon} {layer}"):
            st.write(desc)
    
    st.markdown("---")
    
    # 导航指南
    st.markdown("## 🧭 Navigation Guide")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        **📊 System Overview**  
        Comprehensive view of the framework design and key components
        
        **🔒 Privacy & Security**  
        Explore privacy-enhancing technologies and data protection mechanisms
        
        **⚖️ Ethics & Fairness**  
        Understand ethical principles and bias mitigation strategies
        """)
    
    with nav_col2:
        st.markdown("""
        **🤖 Risk Assessment**  
        Interactive demo of the risk prediction system
        
        **📈 Monitoring Dashboard**  
        Real-time fairness metrics and accountability tracking
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>UN SDG 5: Gender Equality</strong></p>
    <p>Eliminate all forms of violence against all women and girls</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
    ⚠️ This is a demonstration system. Always involve trained professionals in actual DV risk assessment.
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
