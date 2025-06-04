from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X,y = load_iris(retrun_X_y=True)
model = RandomForestClssifier()
model.fit(X,y)

joblib.dump(model, 'model.pkl')