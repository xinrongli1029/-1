import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import time

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

# 定义特征名称
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Streamlit的用户界面
st.title("Heart Disease Predictor")

# 用户输入
age = st.number_input("Age:", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], 
                   format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), 
                  format_func=lambda x: cp_options[x])
trestbps = st.number_input("Resting blood pressure (trestbps):", 
                           min_value=50, max_value=200, value=120)
chol = st.number_input("Serum cholestoral in mg/dl (chol):", 
                       min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], 
                   format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')
restecg = st.selectbox("Resting electrocardiographic results:", 
                       options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
thalach = st.number_input("Maximum heart rate achieved (thalach):", 
                          min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], 
                     format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", 
                          min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (slope):", 
                     options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", 
                     min_value=0, max_value=4, value=0)
thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), 
                    format_func=lambda x: thal_options[x])

# 处理输入并进行预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
input_data = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(input_data)[0]
    predicted_proba = model.predict_proba(input_data)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据我们的模型预测，您的心脏疾病风险很高。"
            f"模型预测您患有心脏疾病的可能性为{probability:.1f}%。"
            "建议尽快联系心脏专科医生进行进一步的检查和评估。"
        )
    else:
        advice = (
            f"根据我们的模型预测，您的心脏疾病风险较低。"
            f"模型预测您患有心脏疾病的可能性为{probability:.1f}%。"
            "建议保持健康生活方式并定期体检。"
        )
    st.write(advice)

    # 计算 SHAP 值
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data).values

    # 检查 expected_value 的结构并调整
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        baseline_value = explainer.expected_value[0]  # 多分类问题，取第一个类别的基线值
    else:
        baseline_value = explainer.expected_value  # 回归问题或其他单值问题

    # 生成 SHAP 力图并保存
    timestamp = int(time.time())
    plot_filename = f"shap_force_plot_{timestamp}.png"
    shap.force_plot(
        baseline_value,
        shap_values[0],  # 对第一个样本生成 SHAP 力图
        input_data
    ).savefig(plot_filename, bbox_inches='tight', dpi=300)

    # 显示 SHAP 力图
    st.image(plot_filename)



# 运行Streamlit命令生成网页应用