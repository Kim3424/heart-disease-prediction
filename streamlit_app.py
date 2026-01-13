import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# ── Cấu hình trang ────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán Bệnh Tim - Nhóm Đề Tài 8",
    page_icon="🫀",
    layout="centered"
)

# Load model & scaler (cache để load nhanh)
@st.cache_resource
def load_artifacts():
    model = joblib.load('model/best_model.joblib')
    scaler = joblib.load('model/scaler.joblib')
    return model, scaler

model, scaler = load_artifacts()

THRESHOLD = 0.40

# ── Tiêu đề & thông tin nhóm ──────────────────────────────
st.title("🫀 Dự Đoán Nguy Cơ Bệnh Tim")
st.markdown("""
**Đề tài 8: Phát hiện bệnh tim**  
**Nhóm**: [Tên nhóm của các bạn]  
**Thành viên**: Võ Hoàng Kim Ngân, Nhan Gia Huy, Trần Lê Hiếu Nghĩa  
**Giảng viên hướng dẫn**: Trần Trương Tuấn Phát  

Mô hình tốt nhất: **Naive Bayes** (độ chính xác trên tập test ~86.7%)
""")

# ── Hiệu suất model (tính động từ test set) ───────────────
@st.cache_data
def compute_metrics():
    # Load data tương tự Colab để tính metrics
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(url, names=columns)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split đơn giản (test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale (dùng scaler đã load, giả sử fit tương tự)
    X_test_s = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba[:, 1])
    loss = log_loss(y_test, y_proba)
    
    return acc, prec, rec, f1, auc, loss

acc, prec, rec, f1, auc, loss = compute_metrics()

st.subheader("Hiệu Suất Model (Từ Test Set)")
st.markdown(f"""
- **Accuracy**: {acc:.2%}  
- **Precision (macro)**: {prec:.2%}  
- **Recall (macro)**: {rec:.2%}  
- **F1-Score (macro)**: {f1:.2%}  
- **ROC-AUC**: {auc:.2%}  
- **Test Loss (log_loss)**: {loss:.4f}
""")

# ── Form nhập liệu ────────────────────────────────────────
with st.form("patient_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Tuổi", 20, 80, 50)
        sex = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
        cp = st.slider("Loại đau ngực (cp)", 0, 3, 0)
        trestbps = st.slider("Huyết áp nghỉ (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 400, 240)
    
    with col2:
        fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl? (fbs)", [0, 1])
        restecg = st.slider("Kết quả điện tâm đồ nghỉ (restecg)", 0, 2, 1)
        thalach = st.slider("Nhịp tim tối đa đạt được (thalach)", 70, 210, 150)
        exang = st.selectbox("Đau thắt ngực do gắng sức? (exang)", [0, 1])
        oldpeak = st.slider("ST depression do gắng sức (oldpeak)", 0.0, 6.0, 1.0, 0.1)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        slope = st.slider("Độ dốc đoạn ST (slope)", 0, 2, 1)
    with col4:
        ca = st.slider("Số mạch máu chính bị hẹp (ca)", 0, 4, 0)
    with col5:
        thal = st.slider("Thalassemia (thal)", 0, 3, 2)
    
    submitted = st.form_submit_button("🔍 Dự đoán", type="primary", use_container_width=True)

# ── Xử lý khi nhấn nút ────────────────────────────────────
if submitted:
    input_array = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale dữ liệu
    input_scaled = scaler.transform(input_array)
    
    # Dự đoán xác suất
    proba = model.predict_proba(input_scaled)[0]
    risk_prob = proba[1]
    
    # Hiển thị kết quả
    st.subheader("Kết quả dự đoán")
    
    col_a, col_b = st.columns([4, 3])
    with col_a:
        st.metric(
            label="Xác suất có bệnh tim",
            value=f"{risk_prob*100:.1f}%",
            delta="NGUY CƠ CAO" if risk_prob >= THRESHOLD else "NGUY CƠ THẤP",
            delta_color="normal" if risk_prob < THRESHOLD else "inverse"
        )
    
    with col_b:
        if risk_prob >= THRESHOLD:
            st.error("⚠️ NGUY CƠ BỆNH TIM\n→ Nên đi khám chuyên khoa tim mạch ngay!")
        else:
            st.success("✅ NGUY CƠ THẤP\n→ Tim có vẻ bình thường (vẫn nên kiểm tra định kỳ)")
    
    st.info("**Lưu ý quan trọng**: Đây chỉ là mô hình tham khảo, KHÔNG thay thế chẩn đoán của bác sĩ!")
