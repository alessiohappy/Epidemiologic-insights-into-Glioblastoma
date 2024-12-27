# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:23:43 2024

@author: aless
"""

# Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss


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


'''#2: ML model'''
# Undersampling dataset 
rus = RandomUnderSampler(sampling_strategy = 0.50)
X_res, y_res = rus.fit_resample(X, y)

print(sorted(Counter(y_res).items()))

# Classification task
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, stratify = y_res)

# Define the objective function to be maximized
def opt_mlp(learning_rate_init, max_iter, hidden_layer_sizes, alpha, beta_1):
    params = {'hidden_layer_sizes': int(hidden_layer_sizes),
              'learning_rate_init': float(learning_rate_init),
              'max_iter': int(max_iter),
              'alpha' : float(alpha),
              'solver' : "adam",
              'activation' : 'logistic',
              'beta_1': int(beta_1),
              'tol' : 0.0001,
              'warm_start' : True,
              'random_state' : 0
      }  
# Execute the cross-validation phase with the given paramters
    model = MLPClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv = 5, scoring = "recall")
    return scores.mean()

# Define the parameter space for optimization
pbounds = {'learning_rate_init': (0.0001, 0.01),
           'max_iter' : (600, 2000),
           'hidden_layer_sizes' : [int(10), int(26)],
           'alpha' : (0.0001, 1),
           'beta_1': (0, 0.9)}

# Create the bayesian optimization object
bayopt = BayesianOptimization(f = opt_mlp,
                              pbounds = pbounds)

# Bayesian Optimization
bayopt.maximize(init_points = 50, n_iter = 100)
print("Best hyperparameters:", bayopt.max)
best_params = bayopt.max['params']


'''3: ML Model'''
# Create a CatBoost model
model = MLPClassifier(hidden_layer_sizes = int(best_params['hidden_layer_sizes']),
                      learning_rate_init = float(best_params['learning_rate_init']),
                      max_iter = int(best_params['max_iter']),
                      alpha = float(best_params['alpha']),
                      beta_1 = int(best_params['beta_1']),
                      solver = "adam",
                      activation = 'logistic',
                      tol = 0.0001,
                      warm_start = True,
                      random_state = 0)

# Fit the CatBoost object to the data
model.fit(X_train, y_train)
    
# Test the model
y_pred = model.predict(X_test)


'''#3: Model evaluation'''
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Classification report
report = classification_report(y_test, y_pred)
print(report)

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv = 4, scoring = 'recall')
train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)
# Plot the learning curve
plt.figure(figsize = (10, 6))
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
train_loss = []
val_loss = []

for epoch in range(1000):  # Numero di epoche
    # Addestra il modello per una epoca
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    
    # Calcola la loss sul training set
    train_loss.append(log_loss(y_train, model.predict_proba(X_train)))
    
    # Calcola la loss sul validation set
    val_loss.append(log_loss(y_test, model.predict_proba(X_test)))

# Plotting della loss durante l'addestramento e la validation
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  

# Confusion matrix
confusion_matrix(y_test, y_pred)
pcm = plot_confusion_matrix(y_test, y_pred)

# Normalized confusion matrix
confusion_matrix(y_test, y_pred)
pncm = plot_confusion_matrix(y_test, y_pred, normalize = True)

# AUC-ROC curve
y_proba = model.predict_proba(X_test)
aucroc = plot_roc(y_test, y_proba, plot_macro = False, plot_micro = False)

# Save the results
with open("mlpBSCV_S1.txt","w") as f:
    f.write("accuracy "+str(accuracy)+"\n")
    f.write("report "+str(report)+"\n")
    f.write("best_param"+str(best_params)+"\n")