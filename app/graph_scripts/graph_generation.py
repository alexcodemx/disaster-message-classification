import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine

def load_data():
    """Load data from the db

    Loads data from the database in a data frame.

    Args:
        None

    Returns:
        df

    """    

    engine = create_engine('sqlite:///data/DisasterResponse.db')

    df = pd.read_sql_table('Messages', engine)

    return df

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # Load data to data frame
    df = load_data()

    # First Graph: top 10 categories

    graph_one = list()

    df_message_cat = df.melt(id_vars='genre', value_vars=df.iloc[:,4:].columns)
    df_message_cat.variable = pd.Series(df_message_cat.variable).replace(r'_',' ',regex=True)
    df_message_cat_counts = df_message_cat.groupby('variable')['value'].sum().sort_values(ascending=False).head(10)
    df_message_cat_names = list(df_message_cat_counts.index.str.title())

    graph_one.append(
        go.Bar(
            x = df_message_cat_names,
            y = df_message_cat_counts,
            marker_color='indianred'
        )
    )

    layout_one = dict(title = 'Top 10 Message Categories',
                xaxis = dict(title = 'Categories',
                  autotick=False),
                yaxis = dict(title = 'Total'),
                )
    
    # Second Graph: Boxplots of message size

    graph_two = list()
    color_list = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)']

    for i, val in enumerate(df.genre.unique()):
        y = df[df['genre'] == val].message.str.len().sort_values().to_list()
        graph_two.append(
            go.Box(y=y, 
                boxpoints=False,
                marker_color='rgb(107,174,214)',
                line_color=color_list[i],
                name=val.title()
            )
        )

    layout_two = dict(title = 'Message Length',
                xaxis = dict(title = 'Message Genre',
                autotick=False),
                yaxis = dict(title = 'Message Size', range=[0,300]),
                )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
        
    return figures