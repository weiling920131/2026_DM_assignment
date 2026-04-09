import numpy as np
from model.utils import onehot_array
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt

# Here is a loss function for linear regression
def MSE(y,y_pred):
     
	return np.mean((y_pred -y)**2)

def MAE(y, y_pred):
    '''
    Mean Absolute Error Loss
    '''
    # TODO
    return mean_absolute_error(y, y_pred)
    # pass

# Here is a loss function for logistic regression
def logloss(y,y_pred):
     
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
     
	return np.mean(-np.log(y_pred)*y - np.log(1-y_pred)*(1-y))

#-----------------------------------------------------------------#

# Here is the evalution function for linear regression
def evaluate_linear_regression(y_true, y_pred, title='Linear Regression Evaluation'):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # TODO 
    mse = MSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
    }

    print(f"=== {title} ===")
    for name, value in metrics.items():
        print(f'{name:<10}: {value:.4f}')

    return metrics

# Here is the evalution function for logistic regression
def evaluate_binary_classifier(y_true, y_pred, title='Model Evaluation'):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # TODO 
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred)
    }

    print(title)
    for name, value in metrics.items():
        print(f'{name:<10}: {value:.4f}')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 4.2))

    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return metrics