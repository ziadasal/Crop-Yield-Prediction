from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pydot

def visualize_tree(data, feature_names, file_name='tree.png'):

    rfr = RandomForestRegressor()
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfr.fit(X_train, y_train)
    tree = rfr.estimators_[5]

    # Export the tree as a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_names, rounded=True, precision=1)
    # Use the dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')

    # Write the graph to a PNG file
    graph.write_png(file_name)
