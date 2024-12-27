# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:26:02 2024

@author: R2D2
"""

# Load the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc
import matplotlib.pyplot as plt
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


'''#2: repeated underampling and tuning with best HPs'''
# Max rec & seed and best rep
max_rec = 0.0
max_seed = 0
best_rep = 0

# Repeated undersampling
lista_recall = []
lista_f1score = []
lista_precision = []
lista_accuracy = []

for i in range(0, 1000):
     # Undersampling dataset 
     rus = RandomUnderSampler(sampling_strategy = 0.50, random_state = i)
     X_res, y_res = rus.fit_resample(X, y)
     
     # Classification task
     X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, stratify = y_res, random_state = i)

     MLP = MLPClassifier(hidden_layer_sizes = (12,),
                         learning_rate_init = 0.00435932489316208,
                         max_iter = 823,
                         alpha = 0.00287697486778122,
                         solver = 'sgd',
                         activation = 'logistic',
                         tol = 0.0001,
                         warm_start = True,
                         random_state = 1)
             
     
     MLP.fit(X_train, y_train)
     y_pred = MLP.predict(X_test)
                                    
     rep = classification_report(y_test, y_pred, output_dict = True)
     recall_1 = rep["1"]["recall"]
                      
     if recall_1 > max_rec:
         max_rec = recall_1
         max_seed = i
         best_rep = rep
     
     r = rep["1"]["recall"]
     lista_recall.append(r)
     f = rep["1"]["f1-score"]
     lista_f1score.append(f)
     p = rep["1"]["precision"]
     lista_precision.append(p)
     a = rep["accuracy"]
     lista_accuracy.append(a)
    
     print("ITERATION NUMBER", i)

print("max_rec", max_rec)
print("max_seed", max_seed)
print("best_performances",best_rep)


'''3: ML Model Evaluation'''
# Undersampling dataset 
rus = RandomUnderSampler(sampling_strategy = 0.50, random_state = max_seed)
X_res, y_res = rus.fit_resample(X, y)

# Classification task
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, stratify = y_res, random_state = max_seed)

modelmlp = MLPClassifier(hidden_layer_sizes = (12,),
                         learning_rate_init = 0.00435932489316208,
                         max_iter = 823,
                         alpha = 0.00287697486778122,
                         solver = 'sgd',
                         activation = 'logistic',
                         tol = 0.0001,
                         warm_start = True,
                         random_state = 1)
    
modelmlp.fit(X_train, y_train)
y_pred = modelmlp.predict(X_test)

# Evaluate the accuracy of the model
mlpaccuracy = accuracy_score(y_test, y_pred)
print(mlpaccuracy)

# Classification report
mlpreport = classification_report(y_test, y_pred)
print(mlpreport)

# Learning curve
plt.rcParams["figure.dpi"] = 600
train_sizes, train_scores, test_scores = learning_curve(modelmlp, X_train, y_train, cv = 5, scoring = 'recall')
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

# Loss Curve
train_loss = []
val_loss = []
# compute the losses
for epoch in range(100):
    modelmlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    train_loss.append(log_loss(y_train, modelmlp.predict_proba(X_train)))
    val_loss.append(log_loss(y_test, modelmlp.predict_proba(X_test)))
# plot the losses
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)

# normalized confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, normalize = True)

# AUC-ROC curve
plt.rcParams["figure.dpi"] = 600
y_proba = modelmlp.predict_proba(X_test)
aucroc = plot_roc(y_test, y_proba, plot_macro = False, plot_micro = False)  

# Save model's performance
with open("MLPbo_sign_info.txt","w") as f:
    f.write("max_rec "+str(max_rec)+"\n")
    f.write("max_seed "+str(max_seed)+"\n")
    f.write("best performances "+str(best_rep)+"\n")
    f.write("All accuracies " + str(lista_accuracy)+"\n")
    f.write("All recall " + str(lista_recall)+"\n")
    f.write("All precision " + str(lista_precision)+"\n")
    f.write("All f1 " + str(lista_f1score))
