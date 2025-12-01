# DV Risk Assessment System

An ethical, privacy-preserving, and bias-mitigated AI framework for Domestic Violence risk prediction, aligned with UN SDG 5: Gender Equality.

## 🌟 Features

- **Privacy-Preserving**: Differential Privacy, Federated Learning, End-to-end Encryption
- **Ethically Aligned**: Non-maleficence, Justice, Human Oversight, Accountability
- **Bias-Mitigated**: Three-stage mitigation pipeline (Pre/In/Post-processing)
- **Explainable**: SHAP analysis, attention visualization, counterfactual explanations
- **Monitored**: Real-time fairness metrics, audit logs, automated alerts

## 🏗️ Architecture

The system implements a six-layer architecture:

1. **Data Security & Governance Layer**: Anonymization, encryption, RBAC
2. **Privacy-Preserving Learning Layer**: DP, FL, SMPC
3. **Fairness-Oriented Pre-Processing Layer**: Reweighing, SMOTE, targeted imputation
4. **Hybrid Deep Learning Core**: BiLSTM + Transformer
5. **Explainability & Human Oversight Layer**: SHAP, attention, human-in-the-loop
6. **Fairness Monitoring & Accountability Layer**: Continuous monitoring, audit logs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dv-risk-assessment.git
cd dv-risk-assessment

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage

### Navigation

- **System Overview**: Comprehensive view of framework design
- **Privacy & Security**: Privacy-enhancing technologies and data protection
- **Ethics & Fairness**: Ethical principles and bias mitigation strategies
- **Risk Assessment**: Interactive risk prediction demo
- **Monitoring Dashboard**: Real-time fairness metrics and accountability tracking

### Example Workflow

1. Navigate to **Risk Assessment** page
2. Input case information (demographics, incident history, behavioral indicators)
3. View risk prediction and recommended actions
4. Examine model explanations (SHAP, attention weights)
5. Monitor fairness metrics in **Monitoring Dashboard**

## 🔒 Privacy & Security

- All sensitive data is encrypted at rest (AES-256)
- API communications use TLS 1.3
- Role-Based Access Control (RBAC) for user management
- Differential Privacy (ε = 1.0) for model training
- Audit logs for all system activities

## ⚖️ Fairness Guarantees

- Continuous monitoring of FPR/FNR across demographic groups
- Target disparity threshold: < 0.10
- Automated retraining triggers when fairness degrades
- Group-specific threshold calibration

## 🤖 Model Details

### Architecture
- **BiLSTM**: Captures short-term temporal dependencies
- **Transformer**: Models long-range relational patterns
- **Attention Mechanism**: Enables interpretability

### Training
- **Loss Function**: Cost-sensitive cross-entropy (FN penalty > FP penalty)
- **Fairness Constraints**: Equal opportunity maximization
- **Regularization**: L2 + Dropout (0.3)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Precision-Recall Curve
- Group-wise FPR/FNR
- Equalized Odds, Equal Opportunity

## 📊 Monitoring & Accountability

All system activities are logged:
- Risk assessments
- Human reviews
- Model updates
- Data access
- Configuration changes

Logs are immutable and retained for 7 years (compliance requirement).

## ⚠️ Important Disclaimers

1. **Demonstration System**: This is an educational demo. Real DV risk assessment requires trained professionals.
2. **No Automated Actions**: The system provides decision support only; final decisions remain with humans.
3. **Context-Specific**: The framework must be adapted to local legal, cultural, and institutional contexts.

## 🔧 Configuration

Key settings can be modified in `config/settings.py`:

```python
PRIVACY_CONFIG = {
    "differential_privacy_epsilon": 1.0,
    "encryption_algorithm": "AES-256",
}

FAIRNESS_THRESHOLDS = {
    "fpr_disparity": 0.1,
    "fnr_disparity": 0.1,
}
```

## 📝 License

This project is provided for educational purposes. Please ensure compliance with local regulations and ethical guidelines when adapting for real-world use.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows ethical AI principles
- Privacy safeguards are maintained
- Fairness is not compromised
- Changes are well-documented

## 📚 References

Based on research addressing UN SDG 5 (Gender Equality), specifically Target 5.2: Eliminate all forms of violence against women and girls.

Key frameworks:
- GDPR (EU General Data Protection Regulation)
- EU AI Act (High-Risk AI Systems)
- WHO Guidelines on Violence Against Women

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**UN SDG 5: Gender Equality**  
*Eliminate all forms of violence against all women and girls*
