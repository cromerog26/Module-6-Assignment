import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('oddsData.csv')
df.replace({'\\N': np.nan}, inplace = True)

df['home_team_score'] = pd.to_numeric(df['score'])
df['away_team_score'] = pd.to_numeric(df['opponentScore'])
df['spread'] = pd.to_numeric(df['spread'])

df['home_team_score'].fillna(df['home_team_score'].median(), inplace = True)
df['away_team_score'].fillna(df['away_team_score'].median(), inplace = True)
df['spread'].fillna(df['spread'].median(), inplace = True)

df['home_cover'] = ((df['score'] - df['opponentScore']) > df['spread']).astype(int)

x = df[['score', 'opponentScore', 'spread']]
y = df['home_cover']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

for depth in range(1, 21):
    clf = DecisionTreeClassifier(max_depth = depth)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'max_depth = {depth} | Test Accuracy: {accuracy:.2f}')

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_test_np = y_test.to_numpy()
X_test_np = x_test.to_numpy()
wrong_pred = np.where(y_pred != y_test_np)[0]

results = pd.DataFrame(X_test_np, columns=['home_score', 'away_score', 'spread'])
results['actual'] = y_test_np
results['predicted'] = y_pred
results['correct'] = results['actual'] == results['predicted']
results['actual_spread'] = results['home_score'] - results['away_score']

correct_preds = results[results['correct']].head(20)
wrong_preds = results[~results['correct']].head(5)

print("\n20 Correct Predictions:\n")
print(correct_preds[['home_score', 'away_score', 'spread']].to_string(index=False))

print("\n5 Wrong Predictions:\n")
print(wrong_preds[['home_score', 'away_score', 'spread', 'actual_spread']].to_string(index=False))

print("\nAccuracy, Precision, Recall & F1 Scores:\n")
accuracy = accuracy_score(y_test_np, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")
precision = precision_score(y_test_np, y_pred)
print(f"Precision: {precision:.2f}")
recall = recall_score(y_test_np, y_pred)
print(f"Recall: {recall:.2f}")
f1 = f1_score(y_test_np, y_pred)
print(f"F1-Score: {f1:.2f}\n")
