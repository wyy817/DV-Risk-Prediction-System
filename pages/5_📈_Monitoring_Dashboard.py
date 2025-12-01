"""
监控仪表板页面
实时公平性指标和问责追踪
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from config.settings import CUSTOM_CSS, DEMOGRAPHIC_GROUPS, FAIRNESS_THRESHOLDS

st.set_page_config(page_title="Monitoring Dashboard", page_icon="📈", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("📈 Monitoring Dashboard")
    st.markdown("### Real-time Fairness & Accountability Tracking")
    
    # 时间范围选择
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        date_range = st.date_input(
            "Analysis Period",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
    
    with col2:
        refresh_rate = st.selectbox(
            "Refresh Rate",
            ["Real-time", "Every 5 minutes", "Hourly", "Daily"]
        )
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # 关键指标卡片
    st.markdown("## Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", "1,247", "+52 today")
    
    with col2:
        st.metric("High Risk Cases", "89", "+3 today", delta_color="inverse")
    
    with col3:
        st.metric("Human Reviews", "156", "+12 today")
    
    with col4:
        st.metric("Fairness Score", "0.92", "+0.02")
    
    st.markdown("---")
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fairness Metrics",
        "🎯 Model Performance",
        "📝 Audit Logs",
        "🚨 Alerts"
    ])
    
    with tab1:
        show_fairness_monitoring()
    
    with tab2:
        show_model_performance()
    
    with tab3:
        show_audit_logs()
    
    with tab4:
        show_alerts()

def show_fairness_monitoring():
    """公平性监控"""
    st.markdown("### Fairness Metrics Across Demographic Groups")
    
    # 生成模拟数据
    groups = ['Overall', 'Age 18-30', 'Age 31-45', 'Age 46-60', 'Age 60+', 
              'Native', 'Migrant', 'Urban', 'Rural', 'High SES', 'Low SES']
    
    metrics_data = {
        'Group': groups,
        'TPR': np.random.uniform(0.75, 0.85, len(groups)),
        'FPR': np.random.uniform(0.08, 0.15, len(groups)),
        'FNR': np.random.uniform(0.10, 0.18, len(groups)),
        'Precision': np.random.uniform(0.70, 0.80, len(groups)),
        'Samples': np.random.randint(50, 200, len(groups))
    }
    
    df = pd.DataFrame(metrics_data)
    
    # 选择指标
    col1, col2 = st.columns(2)
    
    with col1:
        selected_metric = st.selectbox(
            "Select Metric",
            ["FNR", "FPR", "TPR", "Precision"]
        )
    
    with col2:
        threshold = st.slider(
            "Disparity Threshold",
            min_value=0.0,
            max_value=0.2,
            value=float(FAIRNESS_THRESHOLDS.get('fnr_disparity', 0.1)),
            step=0.01
        )
    
    # 可视化
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Group'],
        y=df[selected_metric],
        marker_color=['#2196F3'] + ['#90CAF9'] * (len(groups)-1),
        text=df[selected_metric].round(3),
        textposition='auto',
    ))
    
    # 添加阈值线
    overall_value = df.loc[0, selected_metric]
    fig.add_hline(y=overall_value + threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Upper Bound (+{threshold})")
    fig.add_hline(y=overall_value - threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Lower Bound (-{threshold})")
    
    fig.update_layout(
        title=f"{selected_metric} Across Groups (Target Disparity < {threshold})",
        yaxis_title=selected_metric,
        xaxis_title="Group",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 差异分析
    st.markdown("### Disparity Analysis")
    
    df['Disparity'] = abs(df[selected_metric] - df.loc[0, selected_metric])
    df['Within_Threshold'] = df['Disparity'] <= threshold
    
    col1, col2 = st.columns(2)
    
    with col1:
        compliant = df['Within_Threshold'].sum() - 1  # 减去Overall
        total = len(df) - 1
        st.metric("Groups Within Threshold", f"{compliant}/{total}", 
                  f"{compliant/total*100:.0f}%")
    
    with col2:
        max_disparity = df.loc[1:, 'Disparity'].max()
        max_group = df.loc[df['Disparity'] == max_disparity, 'Group'].values[0]
        st.metric("Maximum Disparity", f"{max_disparity:.3f}", max_group)
    
    # 详细表格
    st.dataframe(
        df[['Group', selected_metric, 'Disparity', 'Within_Threshold', 'Samples']]
        .style.format({selected_metric: '{:.3f}', 'Disparity': '{:.3f}'})
        .background_gradient(subset=['Disparity'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # 时间序列趋势
    st.markdown("---")
    st.markdown("### Fairness Trend Over Time")
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    trend_data = {
        'Date': dates,
        'Max_FNR_Disparity': np.random.uniform(0.05, 0.15, 30),
        'Max_FPR_Disparity': np.random.uniform(0.03, 0.12, 30)
    }
    
    trend_df = pd.DataFrame(trend_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['Max_FNR_Disparity'],
        mode='lines+markers',
        name='FNR Disparity',
        line=dict(color='#f44336')
    ))
    
    fig.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['Max_FPR_Disparity'],
        mode='lines+markers',
        name='FPR Disparity',
        line=dict(color='#ff9800')
    ))
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="green",
                   annotation_text="Target Threshold")
    
    fig.update_layout(
        title="Maximum Disparity Across All Groups",
        yaxis_title="Disparity",
        xaxis_title="Date",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    """模型性能"""
    st.markdown("### Overall Model Performance")
    
    # 混淆矩阵
    col1, col2 = st.columns(2)
    
    with col1:
        confusion_matrix = np.array([
            [850, 45],   # TN, FP
            [52, 300]    # FN, TP
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title="Confusion Matrix (Last 30 Days)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 性能指标
        tp, fp, fn, tn = 300, 45, 52, 850
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        
        fig = go.Figure(go.Bar(
            x=metrics_df['Value'],
            y=metrics_df['Metric'],
            orientation='h',
            marker_color='#2196F3',
            text=metrics_df['Value'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Performance Metrics",
            xaxis=dict(range=[0, 1]),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC曲线
    st.markdown("---")
    st.markdown("### ROC Curve & Precision-Recall Curve")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 生成模拟ROC曲线
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # 模拟曲线
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = 0.89)',
            line=dict(color='#2196F3', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.50)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Precision-Recall曲线
        recall_vals = np.linspace(0, 1, 100)
        precision_vals = 0.9 * (1 - recall_vals) + 0.5
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall_vals,
            y=precision_vals,
            mode='lines',
            name='PR Curve',
            line=dict(color='#4caf50', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 按风险等级的分布
    st.markdown("---")
    st.markdown("### Risk Level Distribution")
    
    risk_dist = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Count': [745, 412, 89],
        'Percentage': [59.8, 33.0, 7.2]
    })
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_dist['Risk Level'],
        values=risk_dist['Count'],
        hole=0.4,
        marker_colors=['#4caf50', '#ff9800', '#f44336'],
        text=risk_dist['Percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='inside'
    )])
    
    fig.update_layout(
        title="Distribution of Risk Assessments",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_audit_logs():
    """审计日志"""
    st.markdown("### System Audit Logs")
    
    # 生成模拟审计日志
    events = ['Risk Assessment', 'Human Review', 'Threshold Update', 'Data Access', 'Model Retrain']
    users = ['social_worker_01', 'data_scientist_02', 'admin_03', 'auditor_04']
    
    log_data = []
    for i in range(50):
        log_data.append({
            'Timestamp': datetime.now() - timedelta(hours=i),
            'Event': np.random.choice(events),
            'User': np.random.choice(users),
            'Case ID': f"DV-2025-{np.random.randint(1000, 9999)}",
            'Action': np.random.choice(['View', 'Update', 'Create', 'Delete']),
            'Result': np.random.choice(['Success', 'Success', 'Success', 'Failed'])
        })
    
    df = pd.DataFrame(log_data)
    
    # 筛选器
    col1, col2, col3 = st.columns(3)
    
    with col1:
        event_filter = st.multiselect("Event Type", events, default=events)
    
    with col2:
        user_filter = st.multiselect("User", users, default=users)
    
    with col3:
        result_filter = st.multiselect("Result", ['Success', 'Failed'], default=['Success', 'Failed'])
    
    # 应用筛选
    filtered_df = df[
        (df['Event'].isin(event_filter)) &
        (df['User'].isin(user_filter)) &
        (df['Result'].isin(result_filter))
    ]
    
    # 显示统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Events", len(filtered_df))
    
    with col2:
        success_rate = (filtered_df['Result'] == 'Success').sum() / len(filtered_df) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        unique_cases = filtered_df['Case ID'].nunique()
        st.metric("Unique Cases", unique_cases)
    
    # 显示日志表格
    st.dataframe(
        filtered_df.sort_values('Timestamp', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # 导出功能
    if st.button("📥 Export Audit Logs"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"audit_logs_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # 活动热图
    st.markdown("---")
    st.markdown("### Activity Heatmap")
    
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day_name()
    
    heatmap_data = df.groupby(['Day', 'Hour']).size().reset_index(name='Count')
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_data.pivot(index='Day', columns='Hour', values='Count').fillna(0)
    heatmap_pivot = heatmap_pivot.reindex(days_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="System Activity by Day and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alerts():
    """警报系统"""
    st.markdown("### System Alerts & Notifications")
    
    # 当前警报
    st.markdown("#### 🚨 Active Alerts")
    
    alerts = [
        {
            "severity": "High",
            "message": "FNR disparity for migrant group exceeds threshold (0.15 > 0.10)",
            "timestamp": datetime.now() - timedelta(hours=2),
            "status": "Active"
        },
        {
            "severity": "Medium",
            "message": "Model accuracy dropped below 85% for rural cases",
            "timestamp": datetime.now() - timedelta(hours=5),
            "status": "Investigating"
        },
        {
            "severity": "Low",
            "message": "Retraining recommended: 30-day threshold reached",
            "timestamp": datetime.now() - timedelta(days=1),
            "status": "Scheduled"
        }
    ]
    
    for alert in alerts:
        severity_colors = {"High": "#f44336", "Medium": "#ff9800", "Low": "#2196F3"}
        
        st.markdown(f"""
        <div style="border-left: 5px solid {severity_colors[alert['severity']]}; 
                    padding: 15px; margin: 10px 0; background-color: #f5f5f5; border-radius: 5px;">
        <strong>{alert['severity']} Priority</strong> - {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}
        <br/>{alert['message']}
        <br/><em>Status: {alert['status']}</em>
        </div>
        """, unsafe_allow_html=True)
    
    # 警报历史
    st.markdown("---")
    st.markdown("#### 📊 Alert History (Last 30 Days)")
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    alert_history = pd.DataFrame({
        'Date': dates,
        'High': np.random.poisson(0.5, 30),
        'Medium': np.random.poisson(1.5, 30),
        'Low': np.random.poisson(3, 30)
    })
    
    fig = go.Figure()
    
    for severity, color in [('High', '#f44336'), ('Medium', '#ff9800'), ('Low', '#2196F3')]:
        fig.add_trace(go.Bar(
            x=alert_history['Date'],
            y=alert_history[severity],
            name=severity,
            marker_color=color
        ))
    
    fig.update_layout(
        title="Alert Frequency by Severity",
        barmode='stack',
        xaxis_title="Date",
        yaxis_title="Number of Alerts",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 警报配置
    st.markdown("---")
    st.markdown("#### ⚙️ Alert Configuration")
    
    with st.expander("Configure Alert Thresholds"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("FNR Disparity Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            st.number_input("Accuracy Drop Threshold (%)", min_value=0, max_value=100, value=85, step=1)
        
        with col2:
            st.number_input("Retraining Interval (days)", min_value=1, max_value=90, value=30, step=1)
            st.checkbox("Email Notifications", value=True)
        
        if st.button("💾 Save Configuration"):
            st.success("Alert configuration updated successfully!")

if __name__ == "__main__":
    main()
