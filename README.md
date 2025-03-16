# Decision Tree Classification with ID3 and C4.5

## 📌 Overview
This project demonstrates the implementation and comparison of two decision tree algorithms: **ID3** (custom implementation) and **C4.5 (using scikit-learn Gini criterion)** on a bank marketing dataset.

It includes:
- A custom-built ID3 algorithm from scratch
- A C4.5 model using scikit-learn's DecisionTreeClassifier
- Preprocessing, encoding, training, and evaluation steps
- Decision tree visualization using `graphviz`

## 📊 Technologies Used
- Python
- Scikit-learn
- NumPy
- Pandas
- Graphviz

## 📁 Project Structure
```
data/
    bank.csv           → Dataset file (you must add it locally)
src/
    ID3.py             → Custom decision tree (ID3) implementation
    C4.5.py            → Scikit-learn decision tree (C4.5 via Gini/Entropy)
requirements.txt       → Python dependencies
```

## ▶️ How to Run

1. Clone the repository:
```
git clone https://github.com/mohammadbaghershahmir/decision-tree-id3-c45-classification.git
```

2. Add the `bank.csv` file into the `data/` folder.

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the scripts:
```bash
cd src
python ID3.py
python C4.5.py
```

## 📈 Sample Output
- Classification report for C4.5
- Confusion matrix
- Printed decision tree for ID3

## 🏷️ Tags
`machine-learning` `decision-tree` `id3` `c4.5` `classification` `scikit-learn`

## 📄 License
MIT
