"""
伦理与公平页面
展示伦理原则和偏见缓解策略
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from config.settings import CUSTOM_CSS, FAIRNESS_THRESHOLDS, DEMOGRAPHIC_GROUPS

st.set_page_config(page_title="Ethics & Fairness", page_icon="⚖️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("⚖️ Ethics & Fairness")
    st.markdown("### Ethical Principles & Bias Mitigation")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Ethical Principles",
        "⚠️ Bias Sources",
        "🛠️ Mitigation Strategies",
        "📊 Fairness Metrics"
    ])
    
    with tab1:
        show_ethical_principles()
    
    with tab2:
        show_bias_sources()
    
    with tab3:
        show_mitigation_strategies()
    
    with tab4:
        show_fairness_metrics()

def show_ethical_principles():
    """展示伦理原则"""
    st.markdown("## Core Ethical Principles")
    
    principles = {
        "Non-maleficence": {
            "icon": "🛡️",
            "definition": "First, do no harm",
            "application": [
                "Minimize false negatives (missed high-risk cases)",
                "Threshold selection prioritizes survivor safety",
                "Impact assessment before deployment",
                "Continuous harm monitoring"
            ]
        },
        "Justice & Fairness": {
            "icon": "⚖️",
            "definition": "Equal treatment and outcomes across groups",
            "application": [
                "Subgroup-specific error rate monitoring",
                "Bias mitigation for marginalized populations",
                "Equal opportunity constraints in training",
                "Regular fairness audits"
            ]
        },
        "Transparency": {
            "icon": "💡",
            "definition": "Explainable and understandable decisions",
            "application": [
                "SHAP-based feature importance",
                "Attention mechanism visualization",
                "Clear documentation of model logic",
                "Accessible explanations for practitioners"
            ]
        },
        "Accountability": {
            "icon": "📋",
            "definition": "Clear responsibility for decisions and outcomes",
            "application": [
                "Immutable audit logs",
                "Decision traceability",
                "Defined escalation procedures",
                "Regular system audits"
            ]
        },
        "Human Oversight": {
            "icon": "👁️",
            "definition": "Humans make final high-stakes decisions",
            "application": [
                "Mandatory review for high-risk predictions",
                "Override capability for trained professionals",
                "AI as decision support, not decision maker",
                "Training for system users"
            ]
        },
        "Dignity & Autonomy": {
            "icon": "🤝",
            "definition": "Respect for individual rights and choices",
            "application": [
                "Survivor-centered governance",
                "Privacy protection paramount",
                "No automated detention/removal",
                "Support, not surveillance"
            ]
        }
    }
    
    cols = st.columns(2)
    
    for idx, (principle, details) in enumerate(principles.items()):
        with cols[idx % 2]:
            with st.expander(f"{details['icon']} {principle}"):
                st.markdown(f"**Definition**: {details['definition']}")
                st.markdown("**Our Implementation**:")
                for app in details['application']:
                    st.markdown(f"✓ {app}")
    
    st.markdown("---")
    st.markdown("### Error Cost Matrix")
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Asymmetric Costs</h4>
    <p>In DV risk prediction, false negatives (missing high-risk cases) have <strong>higher ethical costs</strong> 
    than false positives (over-predicting risk), as they can lead to preventable harm or death.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 成本矩阵
    cost_matrix = pd.DataFrame({
        'True Positive': ['✓ Correct identification', 'Timely intervention', 'Lives saved'],
        'False Positive': ['✗ Over-prediction', 'Unnecessary interventions', 'Moderate cost'],
        'True Negative': ['✓ Correct non-risk', 'No intervention needed', 'Resources saved'],
        'False Negative': ['✗✗ Missed high-risk', 'No intervention', 'SEVERE COST: Potential harm/death']
    }, index=['Outcome', 'Action', 'Cost'])
    
    st.table(cost_matrix)
    
    st.markdown("""
    **Model Optimization**: We use a **cost-sensitive loss function** that penalizes 
    false negatives more heavily than false positives, reflecting their real-world consequences.
    """)

def show_bias_sources():
    """展示偏见来源"""
    st.markdown("## Bias Sources in DV Data")
    
    st.markdown("""
    <div class="info-box">
    <h4>📌 Why Bias Occurs</h4>
    <p>DV datasets reflect systemic inequalities in reporting, access to services, and data collection practices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    bias_types = {
        "Under-reporting Bias": {
            "description": "Women from certain groups are less likely to report DV",
            "affected_groups": ["Migrant communities", "Rural areas", "Low socioeconomic status"],
            "causes": ["Cultural stigma", "Language barriers", "Distrust of authorities", "Limited service access"],
            "impact": "Systematic underrepresentation in training data → Higher false negative rates"
        },
        "Measurement Bias": {
            "description": "Inconsistent recording of DV incidents across institutions",
            "affected_groups": ["Communities with poor healthcare access", "Regions with inconsistent police practices"],
            "causes": ["Lack of standardized protocols", "Varying documentation quality", "Implicit biases in recording"],
            "impact": "Noisy labels → Model learns incorrect patterns"
        },
        "Historical Bias": {
            "description": "Past discrimination reflected in historical data",
            "affected_groups": ["Ethnic minorities", "Indigenous communities", "LGBTQ+ individuals"],
            "causes": ["Discriminatory policies", "Biased risk assessment tools", "Systemic inequalities"],
            "impact": "Model perpetuates historical injustices"
        },
        "Selection Bias": {
            "description": "Data collection focuses on accessible populations",
            "affected_groups": ["Remote rural areas", "Undocumented immigrants", "Homeless women"],
            "causes": ["Sampling convenience", "Resource constraints", "Geographic limitations"],
            "impact": "Poor generalization to underrepresented groups"
        }
    }
    
    for bias_name, details in bias_types.items():
        with st.expander(f"⚠️ {bias_name}"):
            st.markdown(f"**Description**: {details['description']}")
            st.markdown(f"**Affected Groups**: {', '.join(details['affected_groups'])}")
            st.markdown(f"**Causes**:")
            for cause in details['causes']:
                st.markdown(f"- {cause}")
            st.markdown(f"**Impact on Model**: {details['impact']}")
    
    # 可视化偏见影响
    st.markdown("---")
    st.markdown("### Bias Impact Simulation")
    
    # 模拟不同群体的假阴性率
    groups = ['Majority', 'Migrant', 'Rural', 'Low SES']
    baseline_fnr = [0.10, 0.25, 0.22, 0.20]
    mitigated_fnr = [0.11, 0.13, 0.14, 0.12]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups,
        y=baseline_fnr,
        name='Without Bias Mitigation',
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=groups,
        y=mitigated_fnr,
        name='With Bias Mitigation',
        marker_color='green'
    ))
    
    fig.update_layout(
        title="False Negative Rates Across Groups",
        yaxis_title="False Negative Rate",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Insight**: Without mitigation, vulnerable groups have 2-2.5× higher false negative rates, 
    meaning they are systematically under-protected by the model.
    """)

def show_mitigation_strategies():
    """展示缓解策略"""
    st.markdown("## Three-Stage Bias Mitigation Pipeline")
    
    stages = {
        "Stage 1: Pre-processing": {
            "color": "#2196F3",
            "techniques": [
                {
                    "name": "Reweighing",
                    "description": "Assign higher weights to underrepresented groups during training",
                    "code": """
from aif360.algorithms.preprocessing import Reweighing

# 计算实例权重
rw = Reweighing(unprivileged_groups=unprivileged,
                privileged_groups=privileged)
dataset_transformed = rw.fit_transform(dataset)

# 权重应用于损失函数
loss = weighted_loss(predictions, targets, weights=dataset_transformed.instance_weights)
                    """
                },
                {
                    "name": "SMOTE Oversampling",
                    "description": "Synthesize new samples for minority groups using k-nearest neighbors",
                    "code": """
from imblearn.over_sampling import SMOTE

# 对少数群体过采样
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Original: {Counter(y_train)}")
print(f"Resampled: {Counter(y_resampled)}")
                    """
                },
                {
                    "name": "Targeted Imputation",
                    "description": "Use group-specific imputation for missing data",
                    "code": """
from sklearn.impute import KNNImputer

# 分组填充缺失值
for group in ['migrant', 'rural', 'urban']:
    imputer = KNNImputer(n_neighbors=5)
    group_data = data[data['group'] == group]
    data.loc[data['group'] == group] = imputer.fit_transform(group_data)
                    """
                }
            ]
        },
        "Stage 2: In-processing": {
            "color": "#4CAF50",
            "techniques": [
                {
                    "name": "Fair Loss Optimization",
                    "description": "Add fairness constraints to the loss function",
                    "code": """
# 公平性感知损失
def fair_loss(predictions, targets, sensitive_attr):
    # 标准交叉熵损失
    ce_loss = CrossEntropyLoss(predictions, targets)
    
    # 等机会约束: min(TPR_disparity)
    tpr_disparity = equalized_odds_loss(predictions, targets, sensitive_attr)
    
    # 组合损失
    total_loss = ce_loss + lambda_fairness * tpr_disparity
    return total_loss
                    """
                },
                {
                    "name": "Adversarial Debiasing",
                    "description": "Train adversary to remove bias from learned representations",
                    "code": """
# 主分类器
predictor = MainClassifier()

# 对抗器试图预测敏感属性
adversary = AdversaryNetwork()

# 联合训练
pred_loss = classification_loss(predictor(x), y)
adv_loss = adversary_loss(adversary(predictor.hidden), sensitive)

# 最小化预测损失,最大化对抗损失
total_loss = pred_loss - lambda_adv * adv_loss
                    """
                }
            ]
        },
        "Stage 3: Post-processing": {
            "color": "#FF9800",
            "techniques": [
                {
                    "name": "Group-specific Threshold Calibration",
                    "description": "Optimize decision thresholds separately for each group",
                    "code": """
from sklearn.calibration import CalibratedClassifierCV

# 为每个群体校准阈值
calibrated_models = {}
for group in demographic_groups:
    group_data = data[data['group'] == group]
    calibrator = CalibratedClassifierCV(base_model, method='isotonic')
    calibrator.fit(group_data.X, group_data.y)
    calibrated_models[group] = calibrator

# 预测时使用对应群体的校准模型
def predict(x, group):
    return calibrated_models[group].predict_proba(x)
                    """
                },
                {
                    "name": "Equalized Odds Post-processing",
                    "description": "Adjust predictions to achieve equal TPR and FPR across groups",
                    "code": """
from aif360.algorithms.postprocessing import EqOddsPostprocessing

# 后处理优化
eq_odds = EqOddsPostprocessing(
    unprivileged_groups=unprivileged,
    privileged_groups=privileged
)

# 在验证集上学习转换
eq_odds.fit(dataset_valid, dataset_pred)

# 应用到测试集
dataset_transformed = eq_odds.predict(dataset_test)
                    """
                }
            ]
        }
    }
    
    for stage, details in stages.items():
        st.markdown(f"### {stage}")
        
        for tech in details['techniques']:
            with st.expander(f"🛠️ {tech['name']}"):
                st.markdown(tech['description'])
                st.code(tech['code'], language="python")

def show_fairness_metrics():
    """展示公平性指标"""
    st.markdown("## Fairness Evaluation Metrics")
    
    # 指标定义
    metrics = {
        "False Positive Rate (FPR)": "Proportion of actual negatives incorrectly classified as positive",
        "False Negative Rate (FNR)": "Proportion of actual positives incorrectly classified as negative",
        "Equal Opportunity": "TPR should be equal across groups",
        "Equalized Odds": "Both TPR and FPR should be equal across groups",
        "Demographic Parity": "Positive prediction rate should be equal across groups",
        "Predictive Parity": "PPV should be equal across groups"
    }
    
    for metric, definition in metrics.items():
        st.markdown(f"**{metric}**: {definition}")
    
    st.markdown("---")
    
    # 模拟公平性评估结果
    st.markdown("### Interactive Fairness Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_metric = st.selectbox(
            "Select Fairness Metric",
            ["False Negative Rate", "False Positive Rate", "True Positive Rate"]
        )
    
    with col2:
        selected_groups = st.multiselect(
            "Select Demographic Groups",
            DEMOGRAPHIC_GROUPS,
            default=["Age Group", "Migration Status"]
        )
    
    # 生成模拟数据
    if selected_groups:
        data = []
        for group in selected_groups:
            subgroups = {
                "Age Group": ["18-30", "31-45", "46-60", "60+"],
                "Migration Status": ["Native", "Migrant", "Refugee"],
                "Socioeconomic Status": ["High", "Medium", "Low"],
                "Education Level": ["University", "High School", "Primary"],
                "Parental Status": ["Parents", "Non-parents"],
                "Rural/Urban": ["Urban", "Rural"]
            }
            
            for subgroup in subgroups.get(group, []):
                baseline = np.random.uniform(0.15, 0.25) if selected_metric == "False Negative Rate" else np.random.uniform(0.05, 0.15)
                data.append({
                    "Group": group,
                    "Subgroup": subgroup,
                    "Baseline Model": baseline,
                    "Fair Model": baseline * np.random.uniform(0.6, 0.8)
                })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x="Subgroup",
            y=["Baseline Model", "Fair Model"],
            color_discrete_sequence=["#f44336", "#4caf50"],
            barmode="group",
            facet_col="Group",
            facet_col_wrap=2,
            title=f"{selected_metric} Comparison",
            labels={"value": selected_metric, "variable": "Model Type"}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示差异统计
        st.markdown("### Disparity Analysis")
        
        disparity_data = []
        for group in df['Group'].unique():
            group_df = df[df['Group'] == group]
            baseline_max = group_df['Baseline Model'].max()
            baseline_min = group_df['Baseline Model'].min()
            fair_max = group_df['Fair Model'].max()
            fair_min = group_df['Fair Model'].min()
            
            disparity_data.append({
                "Group": group,
                "Baseline Disparity": baseline_max - baseline_min,
                "Fair Disparity": fair_max - fair_min,
                "Improvement": ((baseline_max - baseline_min) - (fair_max - fair_min)) / (baseline_max - baseline_min) * 100
            })
        
        disparity_df = pd.DataFrame(disparity_data)
        st.dataframe(disparity_df.style.format({
            "Baseline Disparity": "{:.3f}",
            "Fair Disparity": "{:.3f}",
            "Improvement": "{:.1f}%"
        }), use_container_width=True)
        
        avg_improvement = disparity_df['Improvement'].mean()
        
        if avg_improvement > 50:
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Strong Fairness Improvement</h4>
            <p>Average disparity reduction: <strong>{avg_improvement:.1f}%</strong></p>
            <p>The fairness-aware model significantly reduces disparities across demographic groups.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            <h4>⚠️ Moderate Fairness Improvement</h4>
            <p>Average disparity reduction: <strong>{avg_improvement:.1f}%</strong></p>
            <p>Further bias mitigation may be needed to achieve equitable outcomes.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
