import matplotlib.pyplot as plt
import networkx as nx

# Define graph
G = nx.DiGraph()

# Add nodes for each component in the flowchart with groupings for readability
G.add_node("Input Data", pos=(0, 10))
G.add_node("Preprocessed", pos=(0, 8))

# Regression Branch
G.add_node("XGBoost Regressor", pos=(-5, 6))
G.add_node("Random Forest Regressor", pos=(-3, 6))
G.add_node("Hyperparameter Tuning (Reg)", pos=(-4, 5))
G.add_node("5-Fold Cross Validation (Reg)", pos=(-4, 4))
G.add_node("Neural Network (Reg)", pos=(-4, 3))
G.add_node("Residual Bias Analysis (Reg)", pos=(-4, 2))
G.add_node("Predicted Outcome (Regression)", pos=(-4, 1))

# Classification Branch
G.add_node("XGBoost Classifier", pos=(0, 6))
G.add_node("Random Forest Classifier", pos=(2, 6))
G.add_node("Hyperparameter Tuning (Class)", pos=(1, 5))
G.add_node("5-Fold Cross Validation (Class)", pos=(1, 4))
G.add_node("Neural Network (Class)", pos=(1, 3))
G.add_node("Adversarial Debiasing (Class)", pos=(1, 2))
G.add_node("True Positive Rate Parity", pos=(0, 1))
G.add_node("Equalized Odds", pos=(2, 1))

# Multiclassification Branch
G.add_node("XGBoost Multi-Classifier", pos=(5, 6))
G.add_node("Random Forest Multi-Classifier", pos=(7, 6))
G.add_node("Hyperparameter Tuning (Multi)", pos=(6, 5))
G.add_node("5-Fold Cross Validation (Multi)", pos=(6, 4))
G.add_node("Neural Network (Multi)", pos=(6, 3))
G.add_node("Adversarial Debiasing (Multi)", pos=(6, 2))
G.add_node("Severity Prediction", pos=(6, 1))

# Add edges between nodes to define the flow
edges = [
    ("Input Data", "Preprocessed"),
    ("Preprocessed", "XGBoost Regressor"), ("Preprocessed", "Random Forest Regressor"),
    ("XGBoost Regressor", "Hyperparameter Tuning (Reg)"), ("Random Forest Regressor", "Hyperparameter Tuning (Reg)"),
    ("Hyperparameter Tuning (Reg)", "5-Fold Cross Validation (Reg)"),
    ("5-Fold Cross Validation (Reg)", "Neural Network (Reg)"),
    ("Neural Network (Reg)", "Residual Bias Analysis (Reg)"),
    ("Residual Bias Analysis (Reg)", "Predicted Outcome (Regression)"),
    
    ("Preprocessed", "XGBoost Classifier"), ("Preprocessed", "Random Forest Classifier"),
    ("XGBoost Classifier", "Hyperparameter Tuning (Class)"), ("Random Forest Classifier", "Hyperparameter Tuning (Class)"),
    ("Hyperparameter Tuning (Class)", "5-Fold Cross Validation (Class)"),
    ("5-Fold Cross Validation (Class)", "Neural Network (Class)"),
    ("Neural Network (Class)", "Adversarial Debiasing (Class)"),
    ("Adversarial Debiasing (Class)", "True Positive Rate Parity"),
    ("Adversarial Debiasing (Class)", "Equalized Odds"),
    
    ("Preprocessed", "XGBoost Multi-Classifier"), ("Preprocessed", "Random Forest Multi-Classifier"),
    ("XGBoost Multi-Classifier", "Hyperparameter Tuning (Multi)"), ("Random Forest Multi-Classifier", "Hyperparameter Tuning (Multi)"),
    ("Hyperparameter Tuning (Multi)", "5-Fold Cross Validation (Multi)"),
    ("5-Fold Cross Validation (Multi)", "Neural Network (Multi)"),
    ("Neural Network (Multi)", "Adversarial Debiasing (Multi)"),
    ("Adversarial Debiasing (Multi)", "Severity Prediction"),
]

# Add edges to graph
G.add_edges_from(edges)

# Define layout based on the 'pos' attribute
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph with node labels
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=8, font_weight="bold", edge_color="gray", linewidths=1, arrowsize=15)
plt.title("Model Prediction Flowchart with Regression, Classification, and Multi-Classification Paths")
plt.show()
