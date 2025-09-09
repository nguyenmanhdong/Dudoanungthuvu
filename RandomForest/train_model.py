import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Đọc dữ liệu
du_lieu = pd.read_csv("data.csv")
du_lieu = du_lieu.drop(columns=["id", "Unnamed: 32"])
du_lieu["diagnosis"] = du_lieu["diagnosis"].map({"M": 0, "B": 1})

# 2. Chọn 5 đặc trưng
dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

X = du_lieu[dac_trung_quan_trong]
y = du_lieu["diagnosis"]

# 3. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Huấn luyện mô hình
mo_hinh = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)
mo_hinh.fit(X_train, y_train)

# 5. Lưu mô hình và danh sách feature
with open("models/cancer_model.pkl", "wb") as f:
    pickle.dump((mo_hinh, dac_trung_quan_trong), f)

print("✅ Mô hình đã được huấn luyện và lưu tại models/cancer_model.pkl")
