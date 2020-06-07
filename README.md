# Disaster Message Classification

The purpose of this project is to classify messages sent during disasters.

## Libraries
* pandas
* numpy
* sqlalchemy
* pickle
* sklearn
* nltk
* joblib
* plotly
* flask


File Structure

```bash
├── app
│   ├── classificationapp
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── templates
│   │       ├── go.html
│   │       └── index.html
│   ├── graph_scripts
│   │   └── graph_generation.py
│   ├── model_scripts
│   │   └── model_data.py
│   └── run.py
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
└── models
    └── train_classifier.py
```

1. App. All Flask application files.
2. Data. CSV Files and wrangling script.
3. Models. Model generation script.

## Steps for running the scripts

Commands must be executed in the root directory.

1. Generate save the data to the database 
```bash
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
2. Generate model
```bash
    python models/train_classifier.py data/DisasterResponse.db models/model.pkl
```
3. Run App
```bash
    python app/run.py
```

