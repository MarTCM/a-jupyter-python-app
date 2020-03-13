import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X , y)

age = input("What is your age? ")
gender = input("What is your gender? ")

if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 0

prediction = model.predict([[age,gender]])[0]
print("We recommend " + prediction)