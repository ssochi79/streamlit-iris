from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 올바르게 데이터 불러오기
X, y = load_iris(return_X_y=True)

# 올바른 모델 클래스 사용
model = RandomForestClassifier()
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'model.pkl')
