import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)



df = pd.read_csv("dataset_sueldos.csv")
df_copy=df.copy()


df_copy = pd.get_dummies(df_copy, columns=["Nivel_Educativo", "Industria","Ubicación","Género"], drop_first=True)


df_copy["Sueldo_Anual"] = np.log1p(df_copy["Sueldo_Anual"])


X,y= remove_labels(df_copy,"Sueldo_Anual")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


clf_tree = RandomForestRegressor(
    random_state=42, 
    n_estimators=500,
    max_depth=4,
    min_samples_split=5,  
    max_features='sqrt')

clf_tree.fit(X_train, y_train)