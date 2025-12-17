import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

matplotlib.use('TkAgg')


# ====================================================================
# QUESTION 1 =========================================================
# ====================================================================

print("="*10)
print("Starting Question 1...")
print("="*10)

def gini_index(p):
    return 1 - np.sum(p ** 2)


def entropy(p):
    p_log2_p = np.where(p > 0, p * np.log2(p), 0)
    return -np.sum(p_log2_p)


probabilities = np.linspace(0.01, 0.99, 50)
data = np.array([probabilities, 1 - probabilities])

gini_values = [gini_index(p) for p in data.T]
entropy_values = [entropy(p) for p in data.T]

plt.figure(figsize=(8,5))

plt.plot(probabilities, gini_values, label="Gini Index", linewidth=3)
plt.plot(probabilities, entropy_values, label="Entropy", linewidth=3)

plt.xlabel("P")
plt.ylabel("Magnitude")
plt.title("Entropy versus Gini Index")

plt.legend()
plt.grid(True)
plt.savefig('entropy_vs_gini_index.png')
# plt.show()

# ====================================================================
# PHASE 2 ============================================================
# ====================================================================

titanic = sns.load_dataset('titanic')
titanic.dropna(how='any', inplace=True)

# ====================================================================
# QUESTION 4 =========================================================
# ====================================================================

print("="*40)
print("Starting Question 4...")
print("="*40)

print("\nBeginning Decision Tree creation...")

numerical_features = titanic.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\nNumerical feature found:", numerical_features)

if 'survived' in numerical_features:
    numerical_features.remove('survived')

print('Features to use:', numerical_features)

X = titanic[numerical_features]
y = titanic.loc[X.index, 'survived']

print(f"Target variable distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=5805,
                                                    stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set target distribution:\n{y_train.value_counts()}")
print(f"\nTest set target distribution:\n{y_test.value_counts()}")

dt_model = DecisionTreeClassifier(random_state=5805)
dt_model.fit(X_train, y_train)

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Difference (Overfitting): {(train_accuracy - test_accuracy)*100:.2f}%")

print("\n" + "="*60)
print("DECISION TREE PARAMETERS AFTER TRAINING")
print("="*60)
params = {
    'ccp_alpha': dt_model.ccp_alpha,
    'class_weight': dt_model.class_weight,
    'criterion': dt_model.criterion,
    'max_depth': dt_model.max_depth,
    'max_features': dt_model.max_features,
    'max_leaf_nodes': dt_model.max_leaf_nodes,
    'min_impurity_decrease': dt_model.min_impurity_decrease,
    'min_samples_leaf': dt_model.min_samples_leaf,
    'min_samples_split': dt_model.min_samples_split,
    'min_weight_fraction_leaf': dt_model.min_weight_fraction_leaf,
    'monotonic_cst': dt_model.monotonic_cst,
    'random_state': dt_model.random_state,
    'splitter': dt_model.splitter
}

for param, value in params.items():
    print(f"{param:30s}: {value}")

print(f"Number of features: {dt_model.n_features_in_}")
print(f"Feature names: {dt_model.feature_names_in_}")
print(f"Number of classes: {dt_model.n_classes_}")
print(f"Classes: {dt_model.classes_}")
print(f"Tree depth: {dt_model.get_depth()}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")

# decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=numerical_features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization (Unpruned)', fontsize=16, fontweight='bold')
plt.tight_layout()
# plt.savefig('decision_tree_titanic.png', dpi=300, bbox_inches='tight')
print("Tree visualization saved as 'decision_tree_titanic.png'")
# plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")
# plt.show()

print(f"""
The model shows a training accuracy of {train_accuracy*100:.2f}% and a test accuracy 
of {test_accuracy*100:.2f}%.

The gap of {(train_accuracy - test_accuracy)*100:.2f}% between training and test accuracy 
indicates that the model is overfitting to the training data. This is expected 
for an unpruned decision tree, as it tends to create very deep trees that 
memorize the training data rather than learning generalizable patterns.

The unpruned tree has likely created splits for every possible combination in 
the training data, achieving perfect or near-perfect training accuracy but 
failing to generalize well to unseen test data.
""")

# ====================================================================
# QUESTION 5 =========================================================
# ====================================================================

print("="*50)
print("Starting Question 5...")
print("="*50)

titanic_5 = sns.load_dataset('titanic')
titanic_5.dropna(how='any', inplace=True)

numerical_features = titanic_5.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'survived' in numerical_features:
    numerical_features.remove('survived')

X = titanic_5[numerical_features]
y = titanic_5.loc[X.index, 'survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=5805,
    stratify=y
)

tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5],
                     'min_samples_split': [20, 30, 40],
                     'min_samples_leaf': [10, 20, 30],
                     'criterion': ['gini', 'entropy', 'log_loss'],
                     'splitter': ['best', 'random'],
                     'max_features': ['sqrt', 'log2']}]

print(f"\nSearching through {np.prod([len(v) for v in tuned_parameters[0].values()])} combinations...")
print(f"Parameters to tune: {list(tuned_parameters[0].keys())}")

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=5805),
    param_grid=tuned_parameters,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=2,
    return_train_score=True
)

print("\nStarting Grid Search...")
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGrid Search completed in {end_time - start_time:.2f} seconds")

print("\nOptimal hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param:25s}: {value}")

print(f"\nBest cross-validation score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# Get the best model
best_model = grid_search.best_estimator_

# Display all parameters of the best model
print("ALL PARAMETERS OF THE BEST (PRUNED) MODEL")
best_params = {
    'ccp_alpha': best_model.ccp_alpha,
    'class_weight': best_model.class_weight,
    'criterion': best_model.criterion,
    'max_depth': best_model.max_depth,
    'max_features': best_model.max_features,
    'max_leaf_nodes': best_model.max_leaf_nodes,
    'min_impurity_decrease': best_model.min_impurity_decrease,
    'min_samples_leaf': best_model.min_samples_leaf,
    'min_samples_split': best_model.min_samples_split,
    'min_weight_fraction_leaf': best_model.min_weight_fraction_leaf,
    'monotonic_cst': best_model.monotonic_cst,
    'random_state': best_model.random_state,
    'splitter': best_model.splitter
}

for param, value in best_params.items():
    print(f"{param:30s}: {value}")

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy_pruned = accuracy_score(y_train, y_train_pred)
test_accuracy_pruned = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy_pruned:.4f} ({train_accuracy_pruned*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy_pruned:.4f} ({test_accuracy_pruned*100:.2f}%)")
print(f"Difference (Train-Test Gap): {(train_accuracy_pruned - test_accuracy_pruned)*100:.2f}%")

# an unpruned model for comparison
unpruned_model = DecisionTreeClassifier(random_state=5805)
unpruned_model.fit(X_train, y_train)
train_accuracy_unpruned = accuracy_score(y_train, unpruned_model.predict(X_train))
test_accuracy_unpruned = accuracy_score(y_test, unpruned_model.predict(X_test))

comparison_df = pd.DataFrame({
    'Model': ['Unpruned', 'Pruned (Best)'],
    'Train Accuracy': [f"{train_accuracy_unpruned:.4f}", f"{train_accuracy_pruned:.4f}"],
    'Test Accuracy': [f"{test_accuracy_unpruned:.4f}", f"{test_accuracy_pruned:.4f}"],
    'Overfitting Gap': [f"{(train_accuracy_unpruned - test_accuracy_unpruned)*100:.2f}%",
                        f"{(train_accuracy_pruned - test_accuracy_pruned)*100:.2f}%"],
    'Tree Depth': [unpruned_model.get_depth(), best_model.get_depth()],
    'Num Leaves': [unpruned_model.get_n_leaves(), best_model.get_n_leaves()]
})
print(comparison_df.to_string(index=False))

improvement = test_accuracy_pruned - test_accuracy_unpruned
overfitting_reduction = (train_accuracy_unpruned - test_accuracy_unpruned) - (train_accuracy_pruned - test_accuracy_pruned)

print(f"Number of features used: {best_model.n_features_in_}")
print(f"Feature names: {list(best_model.feature_names_in_)}")
print(f"Tree depth: {best_model.get_depth()}")
print(f"Number of leaves: {best_model.get_n_leaves()}")
print(f"Number of nodes: {best_model.tree_.node_count}")

# the pruned decision tree
plt.figure(figsize=(20, 12))
plot_tree(
    best_model,
    feature_names=numerical_features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f'Pruned Decision Tree (max_depth={best_model.max_depth}, Test Accuracy={test_accuracy_pruned:.2%})',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
# plt.savefig('pruned_decision_tree.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("Pruned tree visualization saved as 'pruned_decision_tree.png'")
# plt.show()

# comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy Comparison
ax1 = axes[0, 0]
models = ['Unpruned', 'Pruned']
train_accs = [train_accuracy_unpruned, train_accuracy_pruned]
test_accs = [test_accuracy_unpruned, test_accuracy_pruned]
x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
ax1.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim([0.5, 1.0])
for i, v in enumerate(train_accs):
    ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(test_accs):
    ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# 2. Overfitting Gap
ax2 = axes[0, 1]
gaps = [(train_accuracy_unpruned - test_accuracy_unpruned)*100,
        (train_accuracy_pruned - test_accuracy_pruned)*100]
colors = ['red' if g > 5 else 'orange' if g > 2 else 'green' for g in gaps]
ax2.bar(models, gaps, color=colors, alpha=0.7)
ax2.set_ylabel('Overfitting Gap (%)')
ax2.set_title('Overfitting Gap (Train - Test Accuracy)')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
for i, v in enumerate(gaps):
    ax2.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)

# 3. Tree Complexity
ax3 = axes[1, 0]
x = np.arange(len(models))
depths = [unpruned_model.get_depth(), best_model.get_depth()]
leaves = [unpruned_model.get_n_leaves(), best_model.get_n_leaves()]
width = 0.35
ax3.bar(x - width/2, depths, width, label='Tree Depth', alpha=0.8)
ax3.bar(x + width/2, leaves, width, label='Number of Leaves', alpha=0.8)
ax3.set_ylabel('Count')
ax3.set_title('Tree Complexity Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
for i, v in enumerate(depths):
    ax3.text(i - width/2, v + 1, str(v), ha='center', fontsize=9)
for i, v in enumerate(leaves):
    ax3.text(i + width/2, v + 1, str(v), ha='center', fontsize=9)

# 4. Feature Importance
ax4 = axes[1, 1]
top_features = feature_importance.head(5)
ax4.barh(top_features['Feature'], top_features['Importance'], alpha=0.8, color='steelblue')
ax4.set_xlabel('Importance')
ax4.set_title('Top 5 Feature Importances (Pruned Model)')
ax4.invert_yaxis()

plt.tight_layout()
# plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("Comparison analysis saved as 'model_comparison_analysis.png'")
print("="*70)
# plt.show()

improvement = test_accuracy_pruned - test_accuracy_unpruned
overfitting_reduction = (train_accuracy_unpruned - test_accuracy_unpruned) - (train_accuracy_pruned - test_accuracy_pruned)

print(f"""
Pre-pruning constrains the tree's growth by setting limits on parameters like
max_depth, min_samples_split, and min_samples_leaf. This prevents the tree from
creating overly specific rules that memorize the training data.

Key benefits observed:
1. REDUCED OVERFITTING: The gap between train and test accuracy decreased by
   {overfitting_reduction*100:.2f}%, indicating better generalization.

2. IMPROVED INTERPRETABILITY: The pruned tree has only {best_model.get_depth()} levels compared
   to {unpruned_model.get_depth()} in the unpruned version, making it much easier to understand
   and explain.

3. BETTER GENERALIZATION: By avoiding overfitting to noise in the training data,
   the pruned model is more likely to perform consistently on new, unseen data.

4. REDUCED COMPLEXITY: With {best_model.get_n_leaves()} leaves instead of {unpruned_model.get_n_leaves()}, 
   the model is simpler and faster to evaluate.

The optimal parameters found ({grid_search.best_params_}) strike a balance between
model complexity and predictive performance, demonstrating that simpler models can
often outperform more complex ones when properly tuned.
""")

# ====================================================================
# QUESTION 6 =========================================================
# ====================================================================

print("="*60)
print("Starting Question 6...")
print("="*60)

titanic_6 = sns.load_dataset('titanic')

numerical_features = titanic_6.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'survived' in numerical_features:
    numerical_features.remove('survived')

X = titanic_6[numerical_features]
y = titanic_6.loc[X.index, 'survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=5805,
    stratify=y
)

print("\nTraining unpruned tree to extract cost complexity path...")
unpruned_tree = DecisionTreeClassifier(random_state=5805)
unpruned_tree.fit(X_train, y_train)

path = unpruned_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"Number of alpha values to test: {len(ccp_alphas)}")
print(f"Alpha range: [{ccp_alphas[0]:.6f}, {ccp_alphas[-1]:.6f}]")

ccp_alphas = ccp_alphas[:-1]

trees = []
train_scores = []
test_scores = []
tree_depths = []
tree_leaves = []

print(f"\nTraining {len(ccp_alphas)} trees with different alpha values...")

for i, alpha in enumerate(ccp_alphas):
    tree = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    train_scores.append(train_acc)
    test_scores.append(test_acc)
    tree_depths.append(tree.get_depth())
    tree_leaves.append(tree.get_n_leaves())

    if i % max(1, len(ccp_alphas) // 10) == 0:
        print(f"  Alpha={alpha:.6f}: Train={train_acc:.4f}, Test={test_acc:.4f},"
              f"Depth={tree.get_depth()}, Leaves={tree.get_n_leaves()}")

optimal_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[optimal_idx]
optimal_tree = trees[optimal_idx]

print(f"\nOptimal Alpha (ccp_alpha): {optimal_alpha:.8f}")
print(f"Index in path: {optimal_idx} out of {len(ccp_alphas)}")
print(f"\nPerformance with optimal alpha:")
print(f"  Training Accuracy: {train_scores[optimal_idx]:.4f} ({train_scores[optimal_idx]*100:.2f}%)")
print(f"  Test Accuracy: {test_scores[optimal_idx]:.4f} ({test_scores[optimal_idx]*100:.2f}%)")
print(f"  Overfitting Gap: {(train_scores[optimal_idx] - test_scores[optimal_idx])*100:.2f}%")
print(f"  Tree Depth: {tree_depths[optimal_idx]}")
print(f"  Number of Leaves: {tree_leaves[optimal_idx]}")

optimal_params = {
    'ccp_alpha': optimal_tree.ccp_alpha,
    'class_weight': optimal_tree.class_weight,
    'criterion': optimal_tree.criterion,
    'max_depth': optimal_tree.max_depth,
    'max_features': optimal_tree.max_features,
    'max_leaf_nodes': optimal_tree.max_leaf_nodes,
    'min_impurity_decrease': optimal_tree.min_impurity_decrease,
    'min_samples_leaf': optimal_tree.min_samples_leaf,
    'min_samples_split': optimal_tree.min_samples_split,
    'min_weight_fraction_leaf': optimal_tree.min_weight_fraction_leaf,
    'monotonic_cst': optimal_tree.monotonic_cst,
    'random_state': optimal_tree.random_state,
    'splitter': optimal_tree.splitter
}

for param, value in optimal_params.items():
    print(f"{param:30s}: {value}")

print(f"\nTree Structure:")
print(f"  Number of features: {optimal_tree.n_features_in_}")
print(f"  Feature names: {list(optimal_tree.feature_names_in_)}")
print(f"  Number of classes: {optimal_tree.n_classes_}")
print(f"  Classes: {optimal_tree.classes_}")
print(f"  Tree depth: {optimal_tree.get_depth()}")
print(f"  Number of leaves: {optimal_tree.get_n_leaves()}")
print(f"  Number of nodes: {optimal_tree.tree_.node_count}")

unpruned_train_acc = accuracy_score(y_train, unpruned_tree.predict(X_train))
unpruned_test_acc = accuracy_score(y_test, unpruned_tree.predict(X_test))

prepruned_tree = DecisionTreeClassifier(
    random_state=5805,
    max_depth=3,
    min_samples_split=30,
    min_samples_leaf=10
)
prepruned_tree.fit(X_train, y_train)
prepruned_train_acc = accuracy_score(y_train, prepruned_tree.predict(X_train))
prepruned_test_acc = accuracy_score(y_test, prepruned_tree.predict(X_test))

comparison_df = pd.DataFrame({
    'Model': ['Unpruned', 'Pre-Pruned (GridSearch)', 'Post-Pruned (Cost Complexity)'],
    'Train Accuracy': [f"{unpruned_train_acc:.4f}", f"{prepruned_train_acc:.4f}",
                       f"{train_scores[optimal_idx]:.4f}"],
    'Test Accuracy': [f"{unpruned_test_acc:.4f}", f"{prepruned_test_acc:.4f}",
                      f"{test_scores[optimal_idx]:.4f}"],
    'Overfitting Gap': [f"{(unpruned_train_acc - unpruned_test_acc)*100:.2f}%",
                        f"{(prepruned_train_acc - prepruned_test_acc)*100:.2f}%",
                        f"{(train_scores[optimal_idx] - test_scores[optimal_idx])*100:.2f}%"],
    'Tree Depth': [unpruned_tree.get_depth(), prepruned_tree.get_depth(),
                   tree_depths[optimal_idx]],
    'Num Leaves': [unpruned_tree.get_n_leaves(), prepruned_tree.get_n_leaves(),
                   tree_leaves[optimal_idx]]
})

print("\n" + comparison_df.to_string(index=False))

ax1 = axes[0, 0]
ax1.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy',
         linewidth=2, markersize=4, alpha=0.7)
ax1.plot(ccp_alphas, test_scores, marker='s', label='Test Accuracy',
         linewidth=2, markersize=4, alpha=0.7)
ax1.axvline(x=optimal_alpha, color='red', linestyle='--', linewidth=2,
            label=f'Optimal Alpha={optimal_alpha:.6f}')
ax1.scatter([optimal_alpha], [test_scores[optimal_idx]], color='red',
            s=200, zorder=5, edgecolors='black', linewidth=2)
ax1.set_xlabel('Alpha (ccp_alpha)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy vs Cost Complexity Parameter (Alpha)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.0])

# Plot 2: Tree Depth vs Alpha
ax2 = axes[0, 1]
ax2.plot(ccp_alphas, tree_depths, marker='o', color='green',
         linewidth=2, markersize=4, alpha=0.7)
ax2.axvline(x=optimal_alpha, color='red', linestyle='--', linewidth=2,
            label=f'Optimal Alpha={optimal_alpha:.6f}')
ax2.scatter([optimal_alpha], [tree_depths[optimal_idx]], color='red',
            s=200, zorder=5, edgecolors='black', linewidth=2)
ax2.set_xlabel('Alpha (ccp_alpha)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tree Depth', fontsize=12, fontweight='bold')
ax2.set_title('Tree Depth vs Alpha', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Number of Leaves vs Alpha
ax3 = axes[1, 0]
ax3.plot(ccp_alphas, tree_leaves, marker='o', color='purple',
         linewidth=2, markersize=4, alpha=0.7)
ax3.axvline(x=optimal_alpha, color='red', linestyle='--', linewidth=2,
            label=f'Optimal Alpha={optimal_alpha:.6f}')
ax3.scatter([optimal_alpha], [tree_leaves[optimal_idx]], color='red',
            s=200, zorder=5, edgecolors='black', linewidth=2)
ax3.set_xlabel('Alpha (ccp_alpha)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Leaves', fontsize=12, fontweight='bold')
ax3.set_title('Number of Leaves vs Alpha', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Model Comparison Bar Chart
ax4 = axes[1, 1]
models = ['Unpruned', 'Pre-Pruned', 'Post-Pruned']
train_accs = [unpruned_train_acc, prepruned_train_acc, train_scores[optimal_idx]]
test_accs = [unpruned_test_acc, prepruned_test_acc, test_scores[optimal_idx]]
x = np.arange(len(models))
width = 0.35
bars1 = ax4.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
bars2 = ax4.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10)
ax4.legend(fontsize=10)
ax4.set_ylim([0.5, 1.0])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
# plt.savefig('cost_complexity_analysis.png', dpi=300, bbox_inches='tight')
print("\nCost complexity analysis plots saved as 'cost_complexity_analysis.png'")
# plt.show()

# Dedicated plot for Question 6: Finding Optimal Alpha
plt.figure(figsize=(12, 6))

plt.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy', 
         linewidth=2.5, markersize=5, alpha=0.7, color='blue')
plt.plot(ccp_alphas, test_scores, marker='s', label='Test Accuracy', 
         linewidth=2.5, markersize=5, alpha=0.7, color='red')
plt.axvline(x=optimal_alpha, color='green', linestyle='--', linewidth=2, 
            label=f'Optimal Alpha={optimal_alpha:.6f}')
plt.scatter([optimal_alpha], [test_scores[optimal_idx]], color='green', 
            s=300, zorder=5, edgecolors='black', linewidth=2, marker='*')

plt.xlabel('Alpha (ccp_alpha)', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=13, fontweight='bold')
plt.title('Train and Test Accuracy vs Cost Complexity Parameter (Alpha)', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('question6_accuracy_vs_alpha.png', dpi=300, bbox_inches='tight')
# plt.show()

print("\nGenerating post-pruned decision tree visualization...")
plt.figure(figsize=(20, 12))
plot_tree(
    optimal_tree,
    feature_names=numerical_features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f'Post-Pruned Decision Tree (Optimal Alpha={optimal_alpha:.6f}, Test Acc={test_scores[optimal_idx]:.2%})',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
# plt.savefig('post_pruned_decision_tree.png', dpi=300, bbox_inches='tight')
print("Post-pruned tree visualization saved as 'post_pruned_decision_tree.png'")
# plt.show()

test_improvement_vs_unpruned = (test_scores[optimal_idx] - unpruned_test_acc)
test_improvement_vs_prepruned = (test_scores[optimal_idx] - prepruned_test_acc) * 100
overfitting_reduction_unpruned = ((unpruned_train_acc - unpruned_test_acc) -
                                  (train_scores[optimal_idx] - test_scores[optimal_idx]))
overfitting_reduction_prepruned = ((prepruned_train_acc - prepruned_test_acc) -
                                   (train_scores[optimal_idx] - test_scores[optimal_idx])) * 100

print(f"""
COMPARISON: POST-PRUNED vs UNPRUNED TREE
-----------------------------------------
Test Accuracy Change: {test_improvement_vs_unpruned:+.2f}%
Overfitting Gap Reduction: {overfitting_reduction_unpruned:.2f} percentage points
Tree Depth Reduction: {unpruned_tree.get_depth()} → {tree_depths[optimal_idx]} 
                      ({unpruned_tree.get_depth() - tree_depths[optimal_idx]} levels removed)
Leaves Reduction: {unpruned_tree.get_n_leaves()} → {tree_leaves[optimal_idx]} 
                  ({unpruned_tree.get_n_leaves() - tree_leaves[optimal_idx]} leaves pruned)

COMPARISON: POST-PRUNED vs PRE-PRUNED TREE
-------------------------------------------
Test Accuracy Change: {test_improvement_vs_prepruned:+.2f}%
Overfitting Gap Change: {overfitting_reduction_prepruned:+.2f} percentage points
Tree Depth Comparison: {prepruned_tree.get_depth()} (pre) vs {tree_depths[optimal_idx]} (post)
Leaves Comparison: {prepruned_tree.get_n_leaves()} (pre) vs {tree_leaves[optimal_idx]} (post)

POST-PRUNING EFFECTIVENESS
======================================

COST COMPLEXITY PRUNING APPROACH:
   Post-pruning using cost complexity (ccp_alpha) is a bottom-up approach that:
   - First grows a full tree (unpruned)
   - Then systematically removes branches that add minimal predictive value
   - Uses a penalty term (alpha) to balance tree size vs accuracy
   - Optimal alpha={optimal_alpha:.6f} was found through systematic evaluation
""")

# ====================================================================
# QUESTION 7 =========================================================
# ====================================================================

print("="*70)
print("Starting Question 7...")
print("="*70)

titanic_7 = sns.load_dataset('titanic')
titanic_7.dropna(how='any', inplace=True)

numerical_features = titanic_7.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'survived' in numerical_features:
    numerical_features.remove('survived')

X = titanic_7[numerical_features]
y = titanic_7.loc[X.index, 'survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=5805,
    stratify=y
)

print(f"\nTraining Logistic Regression classifier using same test and train from step 4...")

lrc = LogisticRegression(random_state=5805)
lrc.fit(X_train, y_train)

acc = accuracy_score(y_test, lrc.predict(X_test)) * 100
print(f"Logisitc Regression model accuracy: {acc:.2f}")

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Difference (Overfitting): {(train_accuracy - test_accuracy)*100:.2f}%")

# ====================================================================
# QUESTION 8 =========================================================
# ====================================================================

print("="*80)
print("Starting Question 8...")
print("="*80)

from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
roc_auc_score, roc_curve, classification_report, precision_score, f1_score)

titanic_comp = sns.load_dataset('titanic')
titanic_comp.dropna(how='any', inplace=True)

numerical_features = titanic_comp.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'survived' in numerical_features:
    numerical_features.remove('survived')

X = titanic_comp[numerical_features]
y = titanic_comp.loc[X.index, 'survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=5805,
    stratify=y
)

print("\n1. Training Pre-Pruned Decision Tree...")
prepruned_dt = DecisionTreeClassifier(
    random_state=5805,
    max_depth=3,
    min_samples_split=30,
    min_samples_leaf=10,
    criterion='gini'
)
prepruned_dt.fit(X_train, y_train)
prepruned_pred = prepruned_dt.predict(X_test)
prepruned_pred_proba = prepruned_dt.predict_proba(X_test)[:, 1]

print("2. Training Post-Pruned Decision Tree...")
# get the optimal alpha
temp_tree = DecisionTreeClassifier(random_state=5805)
temp_tree.fit(X_train, y_train)
path = temp_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

# optimal alpha
best_alpha = None
best_score = 0
for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    score = accuracy_score(y_test, tree.predict(X_test))
    if score > best_score:
        best_score = score
        best_alpha = alpha

postpruned_dt = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_alpha)
postpruned_dt.fit(X_train, y_train)
postpruned_pred = postpruned_dt.predict(X_test)
postpruned_pred_proba = postpruned_dt.predict_proba(X_test)[:, 1]

print("3. Training Logistic Regression...")
lr_model = LogisticRegression(random_state=5805, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# metrics for all 3 models
models = {
    'Pre-Pruned DT': {
        'predictions': prepruned_pred,
        'probabilities': prepruned_pred_proba,
        'model': prepruned_dt
    },
    'Post-Pruned DT': {
        'predictions': postpruned_pred,
        'probabilities': postpruned_pred_proba,
        'model': postpruned_dt
    },
    'Logistic Regression': {
        'predictions': lr_pred,
        'probabilities': lr_pred_proba,
        'model': lr_model
    }
}

results = {}

for model_name, model_data in models.items():
    pred = model_data['predictions']
    pred_proba = model_data['probabilities']
    
    # metrics
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)
    cm = confusion_matrix(y_test, pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc,
        'Confusion Matrix': cm,
        'ROC Data': roc_curve(y_test, pred_proba)
    }


comparison_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Pre-Pruned DT': [
        f"{results['Pre-Pruned DT']['Accuracy']:.4f}",
        f"{results['Pre-Pruned DT']['Precision']:.4f}",
        f"{results['Pre-Pruned DT']['Recall']:.4f}",
        f"{results['Pre-Pruned DT']['F1-Score']:.4f}",
        f"{results['Pre-Pruned DT']['AUC']:.4f}"
    ],
    'Post-Pruned DT': [
        f"{results['Post-Pruned DT']['Accuracy']:.4f}",
        f"{results['Post-Pruned DT']['Precision']:.4f}",
        f"{results['Post-Pruned DT']['Recall']:.4f}",
        f"{results['Post-Pruned DT']['F1-Score']:.4f}",
        f"{results['Post-Pruned DT']['AUC']:.4f}"
    ],
    'Logistic Regression': [
        f"{results['Logistic Regression']['Accuracy']:.4f}",
        f"{results['Logistic Regression']['Precision']:.4f}",
        f"{results['Logistic Regression']['Recall']:.4f}",
        f"{results['Logistic Regression']['F1-Score']:.4f}",
        f"{results['Logistic Regression']['AUC']:.4f}"
    ]
})

print("\n" + comparison_table.to_string(index=False))

# Confusion matrices
for model_name in ['Pre-Pruned DT', 'Post-Pruned DT', 'Logistic Regression']:
    cm = results[model_name]['Confusion Matrix']
    print(f"\n{model_name}:")
    print("                 Predicted")
    print("                 0    1")
    print(f"Actual 0      {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       1      {cm[1,0]:4d} {cm[1,1]:4d}")
    print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    print(f"Recall (Sensitivity): {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    print(f"Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")


fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# 1. ROC Curves
ax1 = fig.add_subplot(gs[:, 0])

colors = ['blue', 'green', 'red']
for (model_name, color) in zip(['Pre-Pruned DT', 'Post-Pruned DT', 'Logistic Regression'], colors):
    fpr, tpr, _ = results[model_name]['ROC Data']
    auc = results[model_name]['AUC']
    ax1.plot(fpr, tpr, color=color, linewidth=2.5, 
             label=f'{model_name} (AUC = {auc:.4f})')

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.5000)')
ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax1.set_title('ROC Curves Comparison', fontsize=15, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# 2. Metrics Comparison Bar Chart
ax2 = fig.add_subplot(gs[0, 1])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(metrics))
width = 0.25

prepruned_scores = [results['Pre-Pruned DT'][m] for m in metrics]
postpruned_scores = [results['Post-Pruned DT'][m] for m in metrics]
lr_scores = [results['Logistic Regression'][m] for m in metrics]

ax2.bar(x - width, prepruned_scores, width, label='Pre-Pruned DT', alpha=0.8, color='blue')
ax2.bar(x, postpruned_scores, width, label='Post-Pruned DT', alpha=0.8, color='green')
ax2.bar(x + width, lr_scores, width, label='Logistic Regression', alpha=0.8, color='red')

ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Metrics Comparison', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=8, loc='lower right')
ax2.set_ylim([0, 1.0])
ax2.grid(True, alpha=0.3, axis='y')

# 3. Summary statistics table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('tight')
ax3.axis('off')

# Create summary table
table_data = []
table_data.append(['Model', 'Accuracy', 'Recall', 'AUC'])
table_data.append(['Pre-Pruned DT', 
                   f"{results['Pre-Pruned DT']['Accuracy']:.3f}",
                   f"{results['Pre-Pruned DT']['Recall']:.3f}",
                   f"{results['Pre-Pruned DT']['AUC']:.3f}"])
table_data.append(['Post-Pruned DT', 
                   f"{results['Post-Pruned DT']['Accuracy']:.3f}",
                   f"{results['Post-Pruned DT']['Recall']:.3f}",
                   f"{results['Post-Pruned DT']['AUC']:.3f}"])
table_data.append(['Logistic Reg', 
                   f"{results['Logistic Regression']['Accuracy']:.3f}",
                   f"{results['Logistic Regression']['Recall']:.3f}",
                   f"{results['Logistic Regression']['AUC']:.3f}"])

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
table[(1, 0)].set_facecolor('#D9E1F2')
table[(2, 0)].set_facecolor('#E2EFDA')
table[(3, 0)].set_facecolor('#FCE4D6')

ax3.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=10)

# plt.tight_layout()
# plt.savefig('comprehensive_classifier_comparison.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("Comprehensive comparison plot saved as 'comprehensive_classifier_comparison.png'")
# plt.show()

# Additional visualization: Side-by-side confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (model_name, ax) in enumerate(zip(['Pre-Pruned DT', 'Post-Pruned DT', 'Logistic Regression'], axes)):
    cm = results[model_name]['Confusion Matrix']
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=16, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not Survived', 'Survived'], fontsize=9)
    ax.set_yticklabels(['Not Survived', 'Survived'], fontsize=9)

plt.tight_layout()
# plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("Confusion matrices comparison saved as 'confusion_matrices_comparison.png'")
# ============================================================================================
# plt.show()
# ============================================================================================

best_accuracy = max(results.items(), key=lambda x: x[1]['Accuracy'])
best_precision = max(results.items(), key=lambda x: x[1]['Precision'])
best_recall = max(results.items(), key=lambda x: x[1]['Recall'])
best_f1 = max(results.items(), key=lambda x: x[1]['F1-Score'])
best_auc = max(results.items(), key=lambda x: x[1]['AUC'])

print(f"""
METRIC-BY-METRIC ANALYSIS:
-------------------------
- Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['Accuracy']:.4f})
- Best Precision: {best_precision[0]} ({best_precision[1]['Precision']:.4f})
- Best Recall:    {best_recall[0]} ({best_recall[1]['Recall']:.4f})
- Best F1-Score:  {best_f1[0]} ({best_f1[1]['F1-Score']:.4f})
- Best AUC:       {best_auc[0]} ({best_auc[1]['AUC']:.4f})
""")

ranks = {name: [] for name in models.keys()}
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
    sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
    for rank, (model_name, _) in enumerate(sorted_models, 1):
        ranks[model_name].append(rank)

avg_ranks = {name: np.mean(rank_list) for name, rank_list in ranks.items()}
best_overall = min(avg_ranks.items(), key=lambda x: x[1])

print(f"""
Based on evaluation across multiple metrics, 
**{best_overall[0]}** performs best overall with an average rank of {best_overall[1]:.2f}.

JUSTIFICATION:
-------------

1. ACCURACY PERSPECTIVE:
   {best_accuracy[0]} achieves the highest accuracy of {best_accuracy[1]['Accuracy']*100:.2f}%, 
   meaning it correctly classifies the most passengers overall.

2. PRECISION vs RECALL TRADE-OFF:
   - Precision (avoiding false alarms): {best_precision[0]} leads with {best_precision[1]['Precision']*100:.2f}%
   - Recall (catching all positives): {best_recall[0]} leads with {best_recall[1]['Recall']*100:.2f}%
   
   For the Titanic dataset where we want to identify survivors:
   - High recall is critical (don't miss actual survivors)
   - {best_recall[0]} is preferable if recall is prioritized

3. AUC (OVERALL DISCRIMINATIVE ABILITY):
   {best_auc[0]} has the best AUC of {best_auc[1]['AUC']:.4f}, indicating superior 
   ability to distinguish between survivors and non-survivors across all 
   classification thresholds.

4. MODEL CHARACTERISTICS:
""")

for model_name in ['Pre-Pruned DT', 'Post-Pruned DT', 'Logistic Regression']:
    cm = results[model_name]['Confusion Matrix']
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print(f"\n   {model_name}:")
    print(f"   - Correctly identified {tp} survivors out of {tp+fn}")
    print(f"   - Correctly identified {tn} non-survivors out of {tn+fp}")
    print(f"   - Made {fp} false positive errors (predicted survival incorrectly)")
    print(f"   - Made {fn} false negative errors (missed actual survivors)")
