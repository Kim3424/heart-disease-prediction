import streamlit as st
import joblib
import numpy as np

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
