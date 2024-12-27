# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:01:01 2024

@author: R2D2
"""

# Load the libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_curve, auc


'''1: Pre-processing'''
# Import dataset
df = pd.read_csv("path", sep = ",", decimal = ".")
df = df.dropna()

c = df.columns

# Define class, continous and categorical variables 
df_no_ID = df.drop('f.eid', axis = 1)

y = df_no_ID['status']

df_no_Pheno = df_no_ID.drop('status', axis = 1)

cont_var = ["age", "standing_height", "right_hand_grip_strength",
            "number_in_household", "cystatin_c",
            "glycated_haemoglobin_HbA1c", "IGF1",
            "systolic_blood_pressure_AutomatedReading", "weighted"]
    
cat_var = ["qualifications", "ownORrent_accomodation_lived",
           "leisure_social_activities", "current_employment_status",
           "weekly_usage_of_mobile_phone_in_last_3_months"]

other_var = ['sex']
    
df_cat_var = df_no_Pheno.drop(cont_var, axis = 1)
df_cat_var = df_cat_var.drop(other_var, axis = 1)

df_cont_var = df_no_Pheno.drop(cat_var, axis = 1)
df_cont_var = df_cont_var.drop(other_var, axis = 1)

df_other_var = df_no_Pheno.drop(cat_var, axis = 1)
df_other_var = df_other_var.drop(cont_var, axis = 1)

# Preprocessing: dummy variables 
enc = OneHotEncoder()
enc.fit(df_cat_var)
cat_var_encoded = enc.transform(df_cat_var).toarray()

cat_var_name = df_cat_var.columns
new_cat_var_names = enc.get_feature_names_out(cat_var_name)
new_df_cat_var = pd.DataFrame(cat_var_encoded, columns = new_cat_var_names)


# Create a df with normalize variables and cat variables 
new_df = np.concatenate((df_cont_var, df_other_var), axis=1)
new_df_1 = np.concatenate((new_df, new_df_cat_var), axis=1)
new_col_names = ["age", "standing_height", "right_hand_grip_strength",
               "number_in_household", "cystatin_c",
               "glycated_haemoglobin_HbA1c", "IGF1",
               "systolic_blood_pressure_AutomatedReading", "weighted", "sex",
               "qualifications_0.0", "qualifications_1.0", "qualifications_2.0",
               "qualifications_3.0", "qualifications_4.0", "qualifications_5.0", 
               "qualifications_6.0", "ownORrent_accomodation_lived_1.0", 
               "ownORrent_accomodation_lived_2.0",
               "ownORrent_accomodation_lived_3.0", 
               "ownORrent_accomodation_lived_4.0", 
               "ownORrent_accomodation_lived_5.0", 
               "ownORrent_accomodation_lived_6.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_-1.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_0.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_1.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_2.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_3.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_4.0", 
               "weekly_usage_of_mobile_phone_in_last_3_months_5.0", 
               "leisure_social_activities_0.0", "leisure_social_activities_1.0", 
               "leisure_social_activities_2.0", "leisure_social_activities_3.0", 
               "leisure_social_activities_4.0", "leisure_social_activities_5.0",
               "current_employment_status_1.0", "current_employment_status_2.0",
               "current_employment_status_3.0", "current_employment_status_4.0",
               "current_employment_status_5.0", "current_employment_status_6.0",
               "current_employment_status_7.0", "current_employment_status_8.0"]

X = pd.DataFrame(data=new_df_1, columns=new_col_names)


'''2: Bayesian Optimization'''
# Undersampling dataset 
rus = RandomUnderSampler(sampling_strategy = 0.50)
X_res, y_res = rus.fit_resample(X, y)

print(sorted(Counter(y_res).items()))

# Classification task
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, stratify = y_res)


'''Bayesian Optimization'''
def opt_svc(C, gamma):
    params = {
        'C': float(C),
        'gamma' : float(gamma)
        }  
# Execute the cross-validation phase with the given paramters
    model = SVC(**params,
                kernel = "rbf",
                class_weight = "balanced")
    scores = cross_val_score(model, X_train, y_train, cv = 3, scoring = "recall")
    return scores.mean()

# Define the parameter space for optimization
pbounds = {'C': (0.001, 10),
           'gamma' : (0.001, 0.1)}

# Create the bayesian optimization object
bayopt = BayesianOptimization(f = opt_svc,
                              pbounds = pbounds)

# Bayesian Optimization
bayopt.maximize(init_points = 50, n_iter = 100)
print("Best hyperparameters:", bayopt.max)
best_params = bayopt.max['params']


'''3: ML model'''
# Create a SVC model
svc = SVC(C = float(best_params['C']),
          gamma = float(best_params['gamma']),
          kernel = "rbf",
          class_weight = "balanced")

# Fit the SVC object to the data
svc.fit(X_train, y_train)

# Test the model
y_pred = svc.predict(X_test)


'''4: Model evaluation'''
# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Compute the classification report
report = classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(svc, X_train, y_train, cv = 5, scoring = 'recall')
train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)
# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning curve")
plt.xlabel("data dimension")
plt.ylabel("Recall")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha = 0.1, color = "r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
plt.plot(train_sizes, train_scores_mean, 'o-', color = "r", label = "Training Recall")
plt.plot(train_sizes, test_scores_mean, 'o-', color = "g", label = "Validation Recall")
plt.legend(loc = "best")
plt.show()

'''# Loss Function'''
# Calculate accuracy on training and validation sets
train_acc = accuracy_score(y_train, svc.predict(X_train))
val_acc = accuracy_score(y_test, svc.predict(X_test))
print(f'Training accuracy: {train_acc}')
print(f'Validation accuracy: {val_acc}')

# Plotting the training and validation accuracy
plt.plot([1], [train_acc], 'bo', label='Training accuracy')
plt.plot([1], [val_acc], 'ro', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)

# Confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, normalize = True)

# AUC-ROC Curve
# Predict probabilities
y_scores = svc.decision_function(X_test)
# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

with open("SVMbo_S1.txt","w") as f:
    f.write("accuracy "+str(accuracy)+"\n")
    f.write("report "+str(report)+"\n")
    f.write("best params "+str(best_params)+"\n")