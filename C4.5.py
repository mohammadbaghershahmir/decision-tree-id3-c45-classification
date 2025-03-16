import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn import tree

file_path = "E:/Ml-Project/HW2_Shahmir_40221570004/bank.csv"

# خواندن داده‌ها از فایل
df = pd.read_csv(file_path, sep=';')

# نمایش چند سطر از داده‌ها
print(df.head(5))

# تشخیص نام ستون‌ها
print(df.columns)
X = df.drop(["y"],axis=1)  #
y = df["y"]  #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# متغیرهای متنی که نیاز به تبدیل دارند
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
# ایجاد یک Label Encoder برای هر متغیر متنی
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le

clf_id3 = DecisionTreeClassifier(criterion="entropy")
clf_id3.fit(X_train, y_train)

clf_c45 = DecisionTreeClassifier(criterion="gini")
clf_c45.fit(X_train, y_train)

y_pred_id3 = clf_id3.predict(X_test)
report_id3 = classification_report(y_test, y_pred_id3)
matrix_id3 = confusion_matrix(y_test, y_pred_id3)

y_pred_c45 = clf_c45.predict(X_test)
report_c45 = classification_report(y_test, y_pred_c45)
matrix_c45 = confusion_matrix(y_test, y_pred_c45)
s
# برای مدل C4.5
print("C4.5 Classification Report:")
print(report_c45)
print("C4.5 Confusion Matrix:")
print(matrix_c45)
#ساخت درخت با ارتفاع حداکثر 5
# ساخت یک مدل درخت تصمیم با ارتفاع 5
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# آموزش مدل با داده‌های آموزشی
clf.fit(X_train, y_train)

# رسم درخت تصمیم
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X_train.columns,
                                class_names=[str(c) for c in set(y)],
                                filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree_height_5")  # ذخیره نمودار درخت در یک فایل
graph.view("decision_tree_height_5")