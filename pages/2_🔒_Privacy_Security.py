"""
隐私与安全页面
展示隐私保护技术和数据安全机制
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from config.settings import CUSTOM_CSS, PRIVACY_CONFIG

st.set_page_config(page_title="Privacy & Security", page_icon="🔒", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("🔒 Privacy & Security")
    st.markdown("### Privacy-Preserving Technologies & Data Protection")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ Differential Privacy",
        "🌐 Federated Learning",
        "🔐 Encryption & Access Control",
        "📋 Compliance"
    ])
    
    with tab1:
        show_differential_privacy()
    
    with tab2:
        show_federated_learning()
    
    with tab3:
        show_encryption_access()
    
    with tab4:
        show_compliance()

def show_differential_privacy():
    """展示差分隐私"""
    st.markdown("## Differential Privacy (DP)")
    
    st.markdown("""
    <div class="info-box">
    <h4>What is Differential Privacy?</h4>
    <p>A mathematical framework that provides provable privacy guarantees by adding calibrated noise 
    to query results, ensuring that the presence or absence of any individual record has minimal 
    impact on the output.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Parameters")
        epsilon = st.slider(
            "Privacy Budget (ε)",
            min_value=0.1,
            max_value=5.0,
            value=float(PRIVACY_CONFIG["differential_privacy_epsilon"]),
            step=0.1,
            help="Lower ε = stronger privacy but less accuracy"
        )
        
        st.markdown(f"""
        **Current Setting**: ε = {epsilon}
        
        - **ε < 1**: Strong privacy protection
        - **ε = 1**: Recommended balance (current)
        - **ε > 3**: Weaker privacy, higher utility
        
        **Noise Scale**: σ = Δf / ε
        """)
    
    with col2:
        # 可视化噪声影响
        st.markdown("### Noise Addition Visualization")
        
        true_values = np.array([30, 45, 60, 75, 90])
        noise = np.random.laplace(0, 1/epsilon, size=5)
        noisy_values = true_values + noise
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'],
            y=true_values,
            name='True Values',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'],
            y=noisy_values,
            name='DP-Protected Values',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title=f"Impact of DP with ε={epsilon}",
            yaxis_title="Risk Score",
            barmode='group',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Implementation")
    
    st.code("""
from opacus import PrivacyEngine

# 初始化差分隐私引擎
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,  # 噪声倍数
    max_grad_norm=1.0,     # 梯度裁剪
)

# 训练时自动添加DP噪声
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # DP噪声在此步骤添加

# 追踪隐私预算
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy budget spent: ε = {epsilon:.2f}")
    """, language="python")

def show_federated_learning():
    """展示联邦学习"""
    st.markdown("## Federated Learning (FL)")
    
    st.markdown("""
    <div class="info-box">
    <h4>Why Federated Learning?</h4>
    <p>DV data is often distributed across multiple agencies (police, hospitals, social services). 
    FL enables collaborative model training without centralizing sensitive data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FL流程图
    st.markdown("### FL Training Process")
    
    fig = go.Figure()
    
    # 添加节点
    nodes = {
        'Central Server': (0.5, 1),
        'Police Dept': (0.2, 0.5),
        'Hospital': (0.5, 0.5),
        'Social Services': (0.8, 0.5),
    }
    
    # 绘制连接
    for agency, (x, y) in list(nodes.items())[1:]:
        fig.add_trace(go.Scatter(
            x=[0.5, x, 0.5],
            y=[1, y, 1],
            mode='lines+markers',
            line=dict(color='lightblue', width=2),
            marker=dict(size=10),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 添加标签
    for name, (x, y) in nodes.items():
        fig.add_annotation(
            x=x, y=y,
            text=name,
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='white',
            bordercolor='blue',
            borderwidth=2
        )
    
    fig.update_layout(
        title="Federated Learning Architecture",
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=300,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Training Steps
        
        1. **Initialize**: Central server sends global model to agencies
        2. **Local Training**: Each agency trains on its local data
        3. **Aggregation**: Server aggregates model updates (not raw data)
        4. **Update**: New global model distributed to agencies
        5. **Repeat**: Process continues until convergence
        
        **Key Advantage**: Raw data never leaves institutional boundaries
        """)
    
    with col2:
        st.markdown("""
        ### Security Enhancements
        
        - **Secure Aggregation**: Encrypted model updates
        - **Differential Privacy**: Noise added to gradients
        - **Client Selection**: Random sampling to prevent data poisoning
        - **Byzantine-Robust Aggregation**: Detect malicious updates
        
        **Privacy Guarantee**: Individual records cannot be inferred from model updates
        """)
    
    st.code("""
import syft as sy

# 创建虚拟workers代表各机构
police = sy.VirtualWorker(hook, id="police")
hospital = sy.VirtualWorker(hook, id="hospital")
social = sy.VirtualWorker(hook, id="social")

# 分发数据到各worker
data_police = data.send(police)
data_hospital = data.send(hospital)
data_social = data.send(social)

# 联邦训练循环
for epoch in range(epochs):
    # 各机构本地训练
    loss_police = train_local(model, data_police)
    loss_hospital = train_local(model, data_hospital)
    loss_social = train_local(model, data_social)
    
    # 聚合模型参数
    model = federated_averaging([
        model.get().copy(),
        model.get().copy(),
        model.get().copy()
    ])
    """, language="python")

def show_encryption_access():
    """展示加密和访问控制"""
    st.markdown("## Encryption & Access Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Encryption Standards")
        
        encryption_methods = {
            "At Rest": {
                "Algorithm": "AES-256",
                "Key Management": "AWS KMS / HashiCorp Vault",
                "Scope": "Database, file storage"
            },
            "In Transit": {
                "Protocol": "TLS 1.3",
                "Certificate": "X.509",
                "Scope": "API calls, data transfer"
            },
            "In Use": {
                "Method": "Homomorphic Encryption",
                "Library": "TenSEAL",
                "Scope": "Computation on encrypted data"
            }
        }
        
        for enc_type, details in encryption_methods.items():
            with st.expander(f"🔐 {enc_type}"):
                for key, value in details.items():
                    st.markdown(f"**{key}**: {value}")
    
    with col2:
        st.markdown("### Role-Based Access Control (RBAC)")
        
        roles = {
            "Data Scientist": ["Model training", "Read anonymized data"],
            "Social Worker": ["View risk predictions", "Access case history"],
            "Administrator": ["Manage users", "Configure system"],
            "Auditor": ["Read audit logs", "View fairness metrics"],
        }
        
        for role, permissions in roles.items():
            with st.expander(f"👤 {role}"):
                for perm in permissions:
                    st.markdown(f"✓ {perm}")
    
    st.markdown("---")
    st.markdown("### Data Minimization")
    
    minimization_col1, minimization_col2 = st.columns(2)
    
    with minimization_col1:
        st.markdown("""
        **Collected Data**
        - Incident timestamps
        - Severity codes
        - Behavioral indicators
        - Protective factors
        - ~~Full names~~ (removed)
        - ~~Exact addresses~~ (removed)
        - ~~SSN/ID numbers~~ (removed)
        """)
    
    with minimization_col2:
        st.markdown("""
        **Retention Policy**
        - Active cases: 2 years
        - Closed cases: 5 years (archived)
        - Aggregated statistics: Indefinite
        - Audit logs: 7 years (compliance)
        
        **Deletion**: Secure wipe (DoD 5220.22-M standard)
        """)

def show_compliance():
    """展示合规性"""
    st.markdown("## Regulatory Compliance")
    
    compliance_frameworks = {
        "GDPR": {
            "icon": "🇪🇺",
            "principles": [
                "Lawfulness, fairness, transparency",
                "Purpose limitation",
                "Data minimization",
                "Accuracy",
                "Storage limitation",
                "Integrity and confidentiality",
                "Accountability"
            ],
            "our_approach": [
                "Clear consent mechanisms for data use",
                "Explicit purpose specification in data governance",
                "Strict data minimization (only essential features)",
                "Regular data quality audits",
                "Automated deletion after retention period",
                "AES-256 encryption + access controls",
                "Immutable audit logs + DPIA"
            ]
        },
        "EU AI Act": {
            "icon": "⚖️",
            "principles": [
                "High-risk AI classification",
                "Risk management system",
                "Data governance",
                "Transparency & disclosure",
                "Human oversight",
                "Accuracy & robustness",
                "Cybersecurity measures"
            ],
            "our_approach": [
                "Classified as high-risk (public safety)",
                "Multi-layer risk mitigation framework",
                "Privacy-by-design data governance",
                "SHAP explanations + documentation",
                "Mandatory human review for critical decisions",
                "Continuous fairness monitoring + retraining",
                "Penetration testing + vulnerability assessments"
            ]
        },
        "WHO Guidelines": {
            "icon": "🏥",
            "principles": [
                "Survivor safety first",
                "Confidentiality",
                "No retraumatization",
                "Informed consent",
                "Non-discrimination",
                "Professional support",
            ],
            "our_approach": [
                "Risk assessments do not trigger automatic actions",
                "Strict access controls + encrypted storage",
                "Privacy-preserving methods (DP/FL)",
                "Clear explanations of system use",
                "Bias mitigation for marginalized groups",
                "Human-in-the-loop with trained professionals"
            ]
        }
    }
    
    for framework, details in compliance_frameworks.items():
        st.markdown(f"### {details['icon']} {framework}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Requirements**")
            for principle in details['principles']:
                st.markdown(f"- {principle}")
        
        with col2:
            st.markdown("**Our Implementation**")
            for approach in details['our_approach']:
                st.markdown(f"✓ {approach}")
        
        st.markdown("---")
    
    st.markdown("""
    <div class="success-box">
    <h4>✅ Compliance Status</h4>
    <p><strong>GDPR</strong>: Fully compliant with Article 5 principles and Article 35 DPIA requirements</p>
    <p><strong>EU AI Act</strong>: Designed to meet requirements for high-risk AI systems (Articles 8-15)</p>
    <p><strong>WHO</strong>: Aligned with ethical guidelines for DV research and intervention</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
