from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
pd.options.display.max_colwidth = 180
app = Flask(__name__)


data_houses = pd.read_csv("dataset_houses_with_population.csv")
data_apartments = pd.read_csv("dataset_apartments_with_population.csv")
data_sharehouses = pd.read_csv("dataset_sharehouses_with_population.csv")
data_original = pd.concat([data_houses, data_apartments, data_sharehouses])
data_original['id'] = np.arange(start = 1, stop = len(data_original) + 1, step = 1)
data_original = data_original.drop(columns="Unnamed: 0")
data_original.set_index("id", inplace=True)
data_original

data = pd.read_csv("cleaned_data_old.csv", index_col="id")
# data.drop(columns=["population"], inplace=True)
data = data.select_dtypes(exclude=object)

# Return X and y.
def separate_data(data, y_column):
    # Convert the training data to X and y.
    X_data = data.drop([y_column], axis=1)
    y_data = data[y_column]
    return X_data, y_data

X_data, y_data = separate_data(data, "Price")
X_data.columns
y_data

# Split the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1111)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)


def get_r2_score(truths, predictions):
    if len(truths) < 2 or len(predictions) < 2:
        return 0
    else:
        return r2_score(truths, predictions)


def return_listing_table(i, data_original, X_test):
    listing = data_original.loc[X_test.iloc[i].name].copy()
    listing.drop(labels=["Price", "Data from realestate book", "Register number", "Cadastre no.", "Notify about incorrect advertisement", "Kulud suvel/talvel"], inplace=True)
    return listing #.to_frame().to_html()


args = np.arange(len(predictions))
np.random.shuffle(args)

truths = []
truth = 0
predictions_human = []
prediction_human = 0
predictions_model = []
prediction_model = 0
i = 0

current_link = ""


@app.route("/clean")
def reset():
    global i, truth, truths, prediction_human, prediction_model, predictions_human, predictions_model, current_link
    truths = []
    truth = 0
    predictions_human = []
    prediction_human = 0
    predictions_model = []
    prediction_model = 0
    i = 0
    current_link = ""

    return redirect("/")


@app.route('/', methods=["GET", "POST"])
def hello_world():
    global i, truth, truths, prediction_human, prediction_model, predictions_human, predictions_model, current_link

    if request.method == "POST":
        truth = int(y_test.iloc[i])
        truths.append(truth)

        prediction_human = int(request.form['price'])
        predictions_human.append(prediction_human)

        prediction_model = round(predictions[i])
        predictions_model.append(prediction_model)

        print(truths, predictions_human, predictions_model)

        current_link = data_original.loc[X_test.iloc[i].name].Link
        i += 1

    
    # return return_listing_table(i, data_original, X_test) + html_body
    listing = return_listing_table(i, data_original, X_test)
    return render_template('index.html', listing=listing, truth=truth, prediction_human=prediction_human,
                           prediction_model=prediction_model, human_score=get_r2_score(truths, predictions_human),
                           model_score=get_r2_score(truths, predictions_model), link=current_link)

if __name__ == "__main__":
    app.run(debug=True)