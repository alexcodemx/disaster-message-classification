from classificationapp import app
import json, plotly
from flask import render_template, request
from graph_scripts.graph_generation import return_figures
from model_scripts.model_data import model_evaluate

@app.route('/')
@app.route('/index')
def index():
    """Render Index

    Render default page.

    Args:
        None

    Returns:
        None

    """   
    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                        ids=ids,
                        figuresJSON=figuresJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Render Go

    Render go page with the results of the query.

    Args:
        None

    Returns:
        None

    """   
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels, classification_results = model_evaluate(query)

    # Render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )