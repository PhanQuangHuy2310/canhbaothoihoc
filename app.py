import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Academic Status Prediction", layout="wide")

# This class must be defined here so joblib can find it when loading the model
class TextImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.fillna("").astype(str).values.ravel()

@st.cache_resource(show_spinner="Đang tải mô hình mới nhất v3...")
def load_model():
    pipeline = joblib.load('model.pkl')
    return pipeline

@st.cache_data
def load_defaults():
    # Load default numeric values so the user doesn't have to fill in all 40 Att_Subjects
    return pd.read_pickle('defaults.pkl')

def main():
    st.title("🎓 Dự đoán Cảnh báo học vụ (Kaggle Competition)")
    st.markdown("Nhập thông tin sinh viên để dự đoán trạng thái học tập.")
    
    # Load resources
    try:
        model = load_model()
        defaults = load_defaults()
    except Exception as e:
        st.error(f"Error loading model or defaults: {e}. Vui lòng train model trước!")
        return

    # Layout inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cơ bản")
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        age = st.number_input("Tuổi", min_value=15, max_value=60, value=20)
        hometown = st.selectbox("Quê quán", ["Hà Nội", "Hải Dương", "Hưng Yên", "Nam Định", "Hải Phòng", "Quảng Ninh", "Nghệ An", "Thanh Hóa", "Thái Bình", "Khác"])
        admission_mode = st.selectbox("Phương thức xét tuyển", ["Tuyển thẳng", "Thi THPT", "ĐGNL", "Xét học bạ"])
        english_level = st.selectbox("Trình độ Tiếng Anh", ["A1", "A2", "B1", "B2", "IELTS 6.0+"])
        club_member = st.selectbox("Tham gia CLB", ["Yes", "No"])
        tuition_debt = st.number_input("Nợ học phí (VND)", min_value=0, value=0, step=1000000)
        
    with col2:
        st.subheader("Thông tin học thuật & Đánh giá")
        training_score = st.slider("Điểm rèn luyện", 0, 100, 80)
        count_f = st.number_input("Số môn trượt (Count_F)", min_value=0, max_value=50, value=0)
        advisor_notes = st.text_area("Ghi chú của cố vấn (Advisor Notes)", "Sinh viên đi học đầy đủ.")
        personal_essay = st.text_area("Bài luận cá nhân (Personal Essay)", "Tôi mong muốn trở thành một sinh viên giỏi.")

    # Create inference dataframe
    input_data = {
        'Gender': [gender],
        'Age': [age],
        'Hometown': [hometown],
        'Current_Address': ["Không rõ"], # Default
        'Admission_Mode': [admission_mode],
        'English_Level': [english_level],
        'Club_Member': [club_member],
        'Tuition_Debt': [tuition_debt],
        'Count_F': [count_f],
        'Training_Score_Mixed': [training_score],
        'Advisor_Notes': [advisor_notes],
        'Personal_Essay': [personal_essay]
    }
    
    # Fill remaining columns with defaults
    df_pred = pd.DataFrame(input_data)
    for col, val in defaults.items():
        if col not in df_pred.columns:
            df_pred[col] = val
            
    if st.button("Dự đoán", type="primary"):
        with st.spinner("Đang dự đoán..."):
            prediction = model.predict(df_pred)[0]
            probability = model.predict_proba(df_pred)[0]
            
        st.markdown("---")
        # model.classes_ gives us the actual class labels in the same order as predict_proba
        classes = model.classes_
        pred_idx = list(classes).index(prediction)
        top_prob = probability[pred_idx] * 100
        
        label_map = {0: "✅ Bình thường", 1: "⚠️ Cảnh báo học vụ", 2: "❌ Bỏ học"}
        label = label_map.get(int(prediction), str(prediction))
        
        if prediction == 0:
            st.success(f"{label}")
        elif prediction == 1:
            st.warning(f"{label}")
        else:
            st.error(f"{label}")
            
        st.metric("Xác suất dự đoán", f"{top_prob:.1f}%")
        
        # Show full probability breakdown
        st.markdown("**Phân phối xác suất toàn bộ:**")
        prob_df = pd.DataFrame({"Trạng thái": [label_map.get(int(c), str(c)) for c in classes], "Xác suất": [f"{p*100:.1f}%" for p in probability]})
        st.dataframe(prob_df, hide_index=True)
        
        with st.expander("📝 Xem dữ liệu đầu vào"):
            st.json(input_data)

if __name__ == "__main__":
    main()
