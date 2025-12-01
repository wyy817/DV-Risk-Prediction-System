"""
系统配置文件
包含页面配置、样式设置和全局常量
"""

# Streamlit页面配置
PAGE_CONFIG = {
    "page_title": "DV Risk Assessment System",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# 自定义CSS样式
CUSTOM_CSS = """
<style>
    /* 主容器样式 */
    .main {
        padding: 0rem 1rem;
    }
    
    /* 信息框样式 */
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 20px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 20px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }
    
    .danger-box {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 20px 0;
    }
    
    /* 特性卡片样式 */
    .feature-card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-card h3 {
        color: #1976D2;
        margin-top: 0;
    }
    
    .feature-card ul {
        padding-left: 20px;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .metric-card p {
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    /* 风险等级标签 */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .risk-low {
        background-color: #4caf50;
        color: white;
    }
    
    .risk-medium {
        background-color: #ff9800;
        color: white;
    }
    
    .risk-high {
        background-color: #f44336;
        color: white;
    }
    
    /* 表格样式 */
    .dataframe {
        font-size: 0.9em;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        background-color: #1976D2;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1565C0;
        border-color: #1565C0;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
"""

# 系统常量
RISK_LEVELS = {
    "low": {"label": "Low Risk", "color": "#4caf50", "threshold": 0.3},
    "medium": {"label": "Medium Risk", "color": "#ff9800", "threshold": 0.7},
    "high": {"label": "High Risk", "color": "#f44336", "threshold": 1.0}
}

# 模型参数
MODEL_CONFIG = {
    "lstm_units": 128,
    "transformer_heads": 8,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
}

# 隐私保护参数
PRIVACY_CONFIG = {
    "differential_privacy_epsilon": 1.0,
    "encryption_algorithm": "AES-256",
    "anonymization_method": "k-anonymity",
    "k_value": 5,
}

# 公平性指标阈值
FAIRNESS_THRESHOLDS = {
    "fpr_disparity": 0.1,  # False Positive Rate差异阈值
    "fnr_disparity": 0.1,  # False Negative Rate差异阈值
    "equal_opportunity_threshold": 0.8,
}

# 人口统计学分组
DEMOGRAPHIC_GROUPS = [
    "Age Group",
    "Socioeconomic Status",
    "Migration Status",
    "Education Level",
    "Parental Status",
    "Rural/Urban",
]

# 特征列表（用于演示）
FEATURE_CATEGORIES = {
    "Incident History": [
        "Number of Previous Incidents",
        "Severity of Past Incidents",
        "Escalation Pattern",
        "Time Since Last Incident",
    ],
    "Behavioral Indicators": [
        "Controlling Behavior",
        "Threats of Violence",
        "Substance Abuse",
        "Jealousy/Possessiveness",
    ],
    "Contextual Factors": [
        "Economic Stress",
        "Social Isolation",
        "Mental Health Concerns",
        "Access to Weapons",
    ],
    "Protective Factors": [
        "Support Network Strength",
        "Economic Independence",
        "Legal Protection Orders",
        "Safety Planning",
    ],
}

# SDG 5目标
SDG5_TARGETS = {
    "5.1": "End all forms of discrimination against women and girls",
    "5.2": "Eliminate all forms of violence against women and girls",
    "5.3": "Eliminate harmful practices",
    "5.4": "Value unpaid care and domestic work",
    "5.5": "Ensure full participation in leadership and decision-making",
    "5.6": "Universal access to reproductive rights",
}
