# ❤️ Phát hiện bệnh tim (Heart Disease Detection)

## 📌 Giới thiệu

Dự án này xây dựng một hệ thống **phát hiện bệnh tim** dựa trên dữ liệu *Cleveland Heart Disease* (UCI Machine Learning Repository), sử dụng nhiều mô hình **Machine Learning** và **Deep Learning** để so sánh hiệu năng và minh họa quy trình từ tiền xử lý → huấn luyện → đánh giá → lưu model để triển khai.

Notebook chính: `HeartDetection.ipynb`

---

## 🎯 Mục tiêu

* Tiền xử lý dữ liệu bệnh tim một cách tự động
* Huấn luyện và so sánh nhiều mô hình học máy
* Đánh giá mô hình bằng các chỉ số chuẩn (Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss)
* Trực quan hóa Confusion Matrix, Loss & Accuracy curves
* Lưu model và scaler để phục vụ triển khai (Streamlit)

---

## 📊 Dataset

* **Tên**: Cleveland Heart Disease Dataset
* **Nguồn**: UCI Machine Learning Repository
* **Số đặc trưng**: 13 đặc trưng đầu vào
* **Nhãn**: Có / Không mắc bệnh tim

Dataset được tải trực tiếp từ UCI bằng URL trong notebook.

---

## ⚙️ Công nghệ sử dụng

* **Ngôn ngữ**: Python 3
* **Thư viện chính**:

  * `numpy`, `pandas`
  * `scikit-learn`
  * `matplotlib`, `seaborn`
  * `tensorflow / keras`
  * `joblib`

---

## 🧠 Các mô hình được sử dụng

Trong notebook, các mô hình sau được triển khai và đánh giá:

1. **Naive Bayes**
2. **Decision Tree**
3. **Support Vector Machine (SVM)**
4. **Multi-Layer Perceptron (MLP – Keras)**

Ngoài ra:

* So sánh hiệu năng các mô hình bằng bảng tổng hợp
* Vẽ Confusion Matrix và Classification Report

---

## 🔁 Quy trình xử lý

1. Import thư viện
2. Tải dữ liệu Cleveland
3. Tiền xử lý dữ liệu:

   * Tự động phát hiện cột nhãn
   * Xử lý giá trị thiếu
4. Chuẩn hóa dữ liệu (StandardScaler)
5. Huấn luyện từng mô hình
6. Đánh giá & so sánh kết quả
7. Trực quan hóa (Confusion Matrix, Loss/Accuracy Curves)
8. Sinh dữ liệu synthetic để kiểm tra độ ổn định
9. Demo dự đoán cho **1 bệnh nhân**
10. Lưu model & scaler để deploy

---

## 📈 Đánh giá mô hình

Các chỉ số được sử dụng:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Log Loss

Confusion Matrix được vẽ cho từng mô hình để phân tích chi tiết.

---

## 💾 Lưu & triển khai

* Model và Scaler được lưu bằng `joblib`
* Có thể sử dụng lại để:

  * Triển khai bằng **Streamlit**
  * Xây dựng Web/App dự đoán bệnh tim

---

## ▶️ Cách chạy project

### 1. Clone repository

```bash
git clone https://github.com/your-username/heart-disease-detection.git
cd heart-disease-detection
```

### 2. Cài đặt thư viện

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow joblib
```

### 3. Chạy notebook

```bash
jupyter notebook HeartDetection.ipynb
```

Hoặc sử dụng **Google Colab**.

---

## 📌 Ghi chú

* Notebook được thiết kế phục vụ **học tập & báo cáo học phần**
* Code có chú thích rõ theo từng cell
* Phù hợp để mở rộng sang Streamlit hoặc Flask

---

## 👩‍💻 Tác giả

**Ngân Võ Hoàng Kim**
Project học tập – Machine Learning / Data Science

---

## ⭐ Gợi ý cải tiến

* Thêm Logistic Regression, Random Forest, XGBoost
* Tuning hyperparameters (GridSearchCV)
* Triển khai giao diện Streamlit hoàn chỉnh
* Đánh giá bằng Cross-Validation

---

> Nếu bạn thấy project hữu ích, hãy ⭐ repository nhé!
