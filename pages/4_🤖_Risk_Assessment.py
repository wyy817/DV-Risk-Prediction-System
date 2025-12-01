"""
风险评估页面
交互式DV风险预测演示
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from config.settings import CUSTOM_CSS, RISK_LEVELS, FEATURE_CATEGORIES

st.set_page_config(page_title="Risk Assessment", page_icon="🤖", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("🤖 DV Risk Assessment")
    st.markdown("### Interactive Risk Prediction Demo")
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Demonstration System</strong>: This is a simulated system for educational purposes. 
    Real DV risk assessment must always involve trained professionals.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "📝 Case Input",
        "🔮 Risk Prediction",
        "💡 Explainability"
    ])
    
    with tab1:
        case_data = show_case_input()
    
    with tab2:
        if case_data:
            risk_result = show_risk_prediction(case_data)
        else:
            st.info("Please complete the case input in the first tab.")
            risk_result = None
    
    with tab3:
        if risk_result:
            show_explainability(case_data, risk_result)
        else:
            st.info("Risk prediction needed before viewing explanations.")

def show_case_input():
    """案例输入界面"""
    st.markdown("## Case Information Input")
    
    # 使用会话状态存储数据
    if 'case_data' not in st.session_state:
        st.session_state.case_data = {}
    
    # 人口统计信息
    with st.expander("👤 Demographic Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.selectbox("Age Group", ["18-30", "31-45", "46-60", "60+"])
            ses = st.selectbox("Socioeconomic Status", ["High", "Medium", "Low"])
        
        with col2:
            migration = st.selectbox("Migration Status", ["Native", "Migrant", "Refugee"])
            education = st.selectbox("Education Level", ["University", "High School", "Primary", "None"])
        
        with col3:
            location = st.selectbox("Location", ["Urban", "Rural"])
            parental = st.selectbox("Parental Status", ["No children", "Has children"])
    
    # 事件历史
    with st.expander("📊 Incident History", expanded=True):
        st.markdown("### Past Incidents")
        
        num_incidents = st.slider("Number of Previous Incidents", 0, 20, 3)
        
        if num_incidents > 0:
            incident_data = []
            for i in range(min(num_incidents, 5)):  # 限制显示最多5个
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        date = st.date_input(
                            f"Incident {i+1} Date",
                            value=datetime.now() - timedelta(days=30*(num_incidents-i)),
                            key=f"date_{i}"
                        )
                    with col2:
                        severity = st.select_slider(
                            f"Severity {i+1}",
                            options=["Minor", "Moderate", "Serious", "Severe"],
                            value="Moderate",
                            key=f"severity_{i}"
                        )
                    with col3:
                        injury = st.checkbox(f"Physical Injury {i+1}", key=f"injury_{i}")
                    
                    incident_data.append({
                        "date": date,
                        "severity": severity,
                        "injury": injury
                    })
            
            # 计算升级模式
            severities = [inc["severity"] for inc in incident_data]
            severity_values = {"Minor": 1, "Moderate": 2, "Serious": 3, "Severe": 4}
            escalation = np.mean([severity_values[s] for s in severities])
        else:
            escalation = 0
    
    # 行为指标
    with st.expander("🚨 Behavioral Indicators"):
        st.markdown("### Warning Signs")
        
        behaviors = {}
        for category, features in FEATURE_CATEGORIES.items():
            if category == "Behavioral Indicators":
                for feature in features:
                    behaviors[feature] = st.slider(
                        feature,
                        min_value=0,
                        max_value=10,
                        value=5,
                        help=f"Rate the severity of {feature.lower()} (0=None, 10=Severe)"
                    )
    
    # 情境因素
    with st.expander("🏠 Contextual Factors"):
        st.markdown("### Risk Amplifiers")
        
        context = {}
        col1, col2 = st.columns(2)
        
        with col1:
            context["economic_stress"] = st.slider("Economic Stress", 0, 10, 5)
            context["social_isolation"] = st.slider("Social Isolation", 0, 10, 4)
        
        with col2:
            context["mental_health"] = st.slider("Mental Health Concerns", 0, 10, 3)
            context["weapon_access"] = st.checkbox("Access to Weapons")
    
    # 保护因素
    with st.expander("🛡️ Protective Factors"):
        st.markdown("### Risk Reducers")
        
        protection = {}
        col1, col2 = st.columns(2)
        
        with col1:
            protection["support_network"] = st.slider("Support Network Strength", 0, 10, 6)
            protection["economic_independence"] = st.slider("Economic Independence", 0, 10, 5)
        
        with col2:
            protection["legal_protection"] = st.checkbox("Legal Protection Order")
            protection["safety_planning"] = st.checkbox("Safety Plan in Place")
    
    # 保存到会话状态
    if st.button("💾 Save Case Data", type="primary"):
        st.session_state.case_data = {
            "demographics": {
                "age": age,
                "ses": ses,
                "migration": migration,
                "education": education,
                "location": location,
                "parental": parental
            },
            "history": {
                "num_incidents": num_incidents,
                "escalation": escalation
            },
            "behaviors": behaviors if 'behaviors' in locals() else {},
            "context": context,
            "protection": protection
        }
        st.success("✅ Case data saved successfully!")
        return st.session_state.case_data
    
    return st.session_state.case_data if st.session_state.case_data else None

def show_risk_prediction(case_data):
    """风险预测"""
    st.markdown("## Risk Assessment Result")
    
    # 模拟风险评分计算
    risk_score = calculate_risk_score(case_data)
    risk_level = determine_risk_level(risk_score)
    
    # 显示结果
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{risk_score:.2f}</h2>
        <p>Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_badge_class = f"risk-{risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""
        <div class="metric-card">
        <span class="risk-badge {risk_badge_class}">{risk_level}</span>
        <p style="margin-top: 15px;">Risk Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # 风险计量表
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#4caf50"},
                    {'range': [30, 70], 'color': "#ff9800"},
                    {'range': [70, 100], 'color': "#f44336"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 推荐行动
    st.markdown("### 🎯 Recommended Actions")
    
    actions = get_recommended_actions(risk_level, case_data)
    
    for priority, action_list in actions.items():
        st.markdown(f"**{priority}**")
        for action in action_list:
            st.markdown(f"- {action}")
    
    # 人工审核提示
    if risk_level in ["Medium Risk", "High Risk"]:
        st.markdown("""
        <div class="danger-box">
        <h4>⚠️ Human Review Required</h4>
        <p>This case requires <strong>mandatory review by a trained professional</strong> 
        before any intervention is initiated. AI assessment serves as decision support only.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 时间序列预测
    st.markdown("---")
    st.markdown("### 📈 Risk Trajectory")
    
    show_risk_trajectory(case_data, risk_score)
    
    return {"score": risk_score, "level": risk_level, "actions": actions}

def show_explainability(case_data, risk_result):
    """可解释性分析"""
    st.markdown("## Model Explanation")
    
    st.markdown("""
    <div class="info-box">
    <h4>💡 Transparency & Interpretability</h4>
    <p>Understanding <strong>why</strong> the model made this prediction is essential for 
    accountability and trust in high-stakes decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SHAP特征重要性
    st.markdown("### Feature Importance (SHAP Values)")
    
    feature_importance = calculate_feature_importance(case_data)
    
    fig = go.Figure(go.Bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        marker=dict(
            color=list(feature_importance.values()),
            colorscale='RdYlGn_r',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Top Contributing Factors",
        xaxis_title="SHAP Value (Impact on Risk)",
        yaxis_title="Features",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 注意力权重
    st.markdown("---")
    st.markdown("### Temporal Attention Weights")
    
    st.markdown("""
    The model uses **attention mechanisms** to focus on the most relevant historical incidents 
    when predicting current risk.
    """)
    
    if case_data['history']['num_incidents'] > 0:
        attention_weights = np.random.dirichlet(np.ones(min(case_data['history']['num_incidents'], 5)))
        
        fig = go.Figure(data=[go.Scatter(
            x=list(range(1, len(attention_weights)+1)),
            y=attention_weights,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=10)
        )])
        
        fig.update_layout(
            title="Attention on Past Incidents",
            xaxis_title="Incident Number (Chronological)",
            yaxis_title="Attention Weight",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Interpretation**: The model pays most attention to Incident {np.argmax(attention_weights)+1}, 
        suggesting it is the most informative for current risk assessment.
        """)
    
    # 反事实解释
    st.markdown("---")
    st.markdown("### Counterfactual Explanations")
    
    st.markdown("""
    **What would need to change to lower the risk level?**
    """)
    
    counterfactuals = [
        "Reduce 'Controlling Behavior' score from 8 to 4",
        "Increase 'Support Network' from 3 to 7",
        "Establish legal protection order",
        "Decrease 'Economic Stress' from 8 to 5"
    ]
    
    for cf in counterfactuals:
        st.markdown(f"- {cf}")
    
    # 置信度区间
    st.markdown("---")
    st.markdown("### Prediction Confidence")
    
    confidence = np.random.uniform(0.75, 0.95)
    
    st.markdown(f"""
    **Model Confidence**: {confidence:.1%}
    
    This represents the model's certainty in its prediction. Higher confidence suggests 
    the case characteristics closely match patterns learned during training.
    """)

def calculate_risk_score(case_data):
    """计算风险评分（模拟）"""
    score = 0.3  # 基准分数
    
    # 事件历史
    score += case_data['history']['num_incidents'] * 0.03
    score += case_data['history']['escalation'] * 0.05
    
    # 行为指标
    if case_data['behaviors']:
        avg_behavior = np.mean(list(case_data['behaviors'].values())) / 10
        score += avg_behavior * 0.3
    
    # 情境因素
    if case_data['context']:
        context_score = (
            case_data['context']['economic_stress'] +
            case_data['context']['social_isolation'] +
            case_data['context']['mental_health']
        ) / 30
        score += context_score * 0.2
        
        if case_data['context']['weapon_access']:
            score += 0.1
    
    # 保护因素（降低风险）
    if case_data['protection']:
        protection_score = (
            case_data['protection']['support_network'] +
            case_data['protection']['economic_independence']
        ) / 20
        score -= protection_score * 0.15
        
        if case_data['protection']['legal_protection']:
            score -= 0.05
        if case_data['protection']['safety_planning']:
            score -= 0.05
    
    return max(0.0, min(1.0, score))

def determine_risk_level(score):
    """确定风险等级"""
    if score < RISK_LEVELS["low"]["threshold"]:
        return "Low Risk"
    elif score < RISK_LEVELS["medium"]["threshold"]:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommended_actions(risk_level, case_data):
    """获取推荐行动"""
    actions = {}
    
    if risk_level == "Low Risk":
        actions["Immediate"] = ["Provide educational materials", "Ensure access to hotline"]
        actions["Follow-up"] = ["Schedule check-in in 3 months", "Monitor situation"]
    
    elif risk_level == "Medium Risk":
        actions["Immediate"] = [
            "Arrange professional counseling",
            "Develop safety plan",
            "Provide emergency contacts"
        ]
        actions["Follow-up"] = [
            "Weekly check-ins for 1 month",
            "Connect with support services",
            "Re-assess risk in 30 days"
        ]
    
    else:  # High Risk
        actions["Immediate (URGENT)"] = [
            "⚠️ Contact crisis intervention team",
            "⚠️ Evaluate need for emergency shelter",
            "⚠️ Consider legal protection order",
            "⚠️ 24/7 safety monitoring"
        ]
        actions["Follow-up"] = [
            "Daily check-ins",
            "Coordinate with law enforcement",
            "Continuous risk monitoring"
        ]
    
    return actions

def calculate_feature_importance(case_data):
    """计算特征重要性（模拟SHAP）"""
    importance = {}
    
    # 随机生成重要性值（实际应使用SHAP）
    if case_data['history']['num_incidents'] > 3:
        importance["Number of Incidents"] = 0.25
    
    if case_data['history']['escalation'] > 2:
        importance["Escalation Pattern"] = 0.20
    
    if case_data['behaviors']:
        top_behavior = max(case_data['behaviors'].items(), key=lambda x: x[1])
        importance[top_behavior[0]] = 0.18
    
    if case_data['context']['economic_stress'] > 7:
        importance["Economic Stress"] = 0.15
    
    if case_data['context']['weapon_access']:
        importance["Weapon Access"] = 0.22
    
    # 标准化
    total = sum(importance.values())
    importance = {k: v/total for k, v in importance.items()}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

def show_risk_trajectory(case_data, current_score):
    """显示风险轨迹"""
    # 模拟历史和未来风险
    months = 12
    past_months = list(range(-months, 0))
    future_months = list(range(0, months+1))
    
    # 生成模拟轨迹
    np.random.seed(42)
    past_risk = current_score - 0.1 + np.cumsum(np.random.randn(months) * 0.02)
    future_risk_baseline = current_score + np.cumsum(np.random.randn(months+1) * 0.02)
    future_risk_intervention = current_score + np.cumsum(np.random.randn(months+1) * 0.015) - 0.02 * np.arange(months+1)
    
    fig = go.Figure()
    
    # 历史风险
    fig.add_trace(go.Scatter(
        x=past_months,
        y=past_risk,
        mode='lines',
        name='Historical Risk',
        line=dict(color='#2196F3', width=2)
    ))
    
    # 当前点
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current_score],
        mode='markers',
        name='Current Risk',
        marker=dict(color='red', size=12)
    ))
    
    # 未来预测（无干预）
    fig.add_trace(go.Scatter(
        x=future_months,
        y=future_risk_baseline,
        mode='lines',
        name='Projected (No Intervention)',
        line=dict(color='#f44336', width=2, dash='dash')
    ))
    
    # 未来预测（有干预）
    fig.add_trace(go.Scatter(
        x=future_months,
        y=future_risk_intervention,
        mode='lines',
        name='Projected (With Intervention)',
        line=dict(color='#4caf50', width=2, dash='dash')
    ))
    
    # 风险阈值线
    fig.add_hline(y=0.7, line_dash="dot", line_color="red", 
                   annotation_text="High Risk Threshold")
    fig.add_hline(y=0.3, line_dash="dot", line_color="orange",
                   annotation_text="Medium Risk Threshold")
    
    fig.update_layout(
        title="12-Month Risk Trajectory",
        xaxis_title="Months (Relative to Current)",
        yaxis_title="Risk Score",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation**: The trajectory shows how risk may evolve with and without intervention. 
    Early intervention can significantly alter the risk trajectory and prevent escalation.
    """)

if __name__ == "__main__":
    main()
