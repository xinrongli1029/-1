import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost.pkl')

# 定义特征的选项
cp_options = {
    1: 'Typical angina (1)',
    2: 'Atypical angina (2)',
    3: 'Non-anginal pain (3)',
    4: 'Asymptomatic (4)'
}

restecg_options = {
    0: 'Normal (0)',
    1: 'ST-T wave abnormality (1)',
    2: 'Left ventricular hypertrophy (2)'
}

slope_options = {
    1: 'Upsloping (1)',
    2: 'Flat (2)',
    3: 'Downsloping (3)'
}

thal_options = {
    3: 'Normal (3)',
    6: 'Fixed defect (6)',
    7: 'Reversible defect (7)'
}

# Streamlit的用户界面
st.title("Heart Disease Predictor")

# age: 数值输入
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: 分类选择
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# cp: 分类选择
cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# trestbps: 数值输入
trestbps = st.number_input("Resting blood pressure (trestbps):", min_value=50, max_value=200, value=120)

# chol: 数值输入
chol = st.number_input("Serum cholestoral in mg/dl (chol):", min_value=100, max_value=600, value=200)

# fbs: 分类选择
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# restecg: 分类选择
restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# thalach: 数值输入
thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)

# exang: 分类选择
exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# oldpeak: 数值输入
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# slope: 分类选择
slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# ca: 数值输入
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)

# thal: 分类选择
thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# 处理输入并进行预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"根据我们的模型预测，您的心脏疾病的风险很高。"
            f"模型预测您患有心脏疾病的可能性为{probability:.1f}%。"
            "虽然这只是一个概率估计，但这表明您可能存在较高的心脏疾病风险。"
            "我建议您尽快联系心脏专科医生进行进一步的检查和评估，"
            "以确保得到准确的诊断和必要的治疗措施。"
        )
    else:
        advice = (
            f"根据我们的模型预测，您的心脏疾病风险较低。"
            f"模型预测您患有心脏疾病的可能性为{probability:.1f}%。"
            "尽管如此，保持健康的生活方式仍然非常重要。"
            "建议您定期进行体检，以监测心脏健康，"
            "并在有任何不适症状时及时就医。"
        )

    st.write(advice)

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=self.feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=self.feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)

    st.image("shap_force_plot.png")

# 运行Streamlit命令生成网页应用