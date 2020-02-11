import numpy as np
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix

def printConfusionMatrix(y_test, y_pred, target_names):
    
    print("|....................Confusion Matrix....................|", 
      "\n\n", 
      tabulate(np.column_stack((target_names, confusion_matrix(y_test, y_pred))), headers=target_names), 
      "\n")
    
def printClassificationReport(y_test, y_pred, target_names):
    
    print("|....................Classification Report....................|", 
      "\n\n", 
      classification_report(y_test, y_pred, target_names=target_names), 
      "\n")