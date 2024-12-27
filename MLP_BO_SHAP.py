# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:16:20 2024

@author: R2D2
"""

# Load the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import shap
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_precision_recall
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.calibration import CalibrationDisplay


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


'''2: ML Model'''
# Undersampling dataset 
rus = RandomUnderSampler(sampling_strategy = 0.50, random_state = 546)
X_res, y_res = rus.fit_resample(X, y)

# Classification task
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, stratify = y_res, random_state = 546)

modelmlp = MLPClassifier(hidden_layer_sizes = (12,),
                         learning_rate_init = 0.004359,
                         max_iter = 823,
                         alpha = 0.002877,
                         solver = 'sgd',
                         activation = 'logistic',
                         tol = 0.0001,
                         warm_start = True,
                         random_state = 1)
    
modelmlp.fit(X_train, y_train)
y_pred = modelmlp.predict(X_test)


'''3: Model evaluation'''
# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# generic report (precision, recall, f1-score and support)
print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred)

# confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)

# normalized confusion matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, normalize = True)

'''# AUC and PR Curves'''
# Predicted probabilities
y_proba = modelmlp.predict_proba(X_test)

# AUC Curve
plt.rcParams['figure.dpi'] = 600
aucroc = plot_roc(y_test, y_proba, plot_macro = False, plot_micro = False)

# PR Curve
plt.rcParams['figure.dpi'] = 600
plot_precision_recall(y_test, y_proba, cmap = "Dark2")
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(modelmlp, X_train, y_train, cv = 5, scoring = 'recall')
train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)
# Plot the learning curve
plt.figure(figsize = (10, 6))
plt.title("Learning curve")
plt.xlabel("Data dimension")
plt.ylabel("Recall")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha = 0.1, color = "r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
plt.plot(train_sizes, train_scores_mean, 'o-', color = "r", label = "Training Recall")
plt.plot(train_sizes, test_scores_mean, 'o-', color = "g", label = "Validation Recall")
plt.legend(loc = "best")  

# calibration from estimation
disp = CalibrationDisplay.from_estimator(modelmlp, X_test, y_test)

'''# Loss Function'''
train_loss = []
val_loss = []

for epoch in range(1000):  # Numero di epoche
    # Addestra il modello per una epoca
    modelmlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    
    # Calcola la loss sul training set
    train_loss.append(log_loss(y_train, modelmlp.predict_proba(X_train)))
    
    # Calcola la loss sul validation set
    val_loss.append(log_loss(y_test, modelmlp.predict_proba(X_test)))

# Plotting della loss durante l'addestramento e la validation
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


'''#4: SHapely Additive exPlanations'''
# Creazione del modello SHAP Explainer
#final_shap_df_v2 = pd.DataFrame(data = X_res, columns = new_col_names)
#shap_values_tree = shap.KernelExplainer(svc.predict, data = X_train).shap_values(X = final_shap_df_v2, y = y_res)

final_shap_df_v2 = pd.read_csv("C:\\Users\\R2D2\\OneDrive - University of Pisa\\"
                               + "UniPi\\OneDrive - University of Pisa\\UniPi\\"
                               + "Articoli\\Glioblastoma\\MLP_articolo_gbm\\"
                               + "final_shap_df_v2_mlp.csv", sep = ",")

final_shap_df_v2 = final_shap_df_v2.drop("Unnamed: 0", axis = 1)

shap_values_tree = np.load("C:\\Users\\R2D2\\OneDrive - University of Pisa\\"
                           + "UniPi\\OneDrive - University of Pisa\\UniPi\\"
                           + "Articoli\\Glioblastoma\\MLP_articolo_gbm\\"
                           + "shap_values_tree_mlp.npy")

print(shap_values_tree)

shap_values_tree_cases = shap_values_tree

# Summary_plot and dependence plot
shap.summary_plot(shap_values_tree, final_shap_df_v2)
shap.summary_plot(shap_values_tree, final_shap_df_v2, plot_type = "bar")

# Split the dataset and shap value in cont and cat variables and keep only cat
X_cat_var = X_res.drop(cont_var, axis = 1)
X_cat_var = X_cat_var.drop(other_var, axis = 1)

shap_values_tree_name = pd.DataFrame(shap_values_tree_cases, columns = new_col_names)

shap_values_categorical = shap_values_tree_name.drop(cont_var, axis = 1)
shap_values_categorical = shap_values_categorical.drop(other_var, axis = 1)

# Get number of unique categories for each feature 
print(new_df_cat_var.columns)
feature_names = X_cat_var.columns
n_categories = [7, 6, 8, 7, 6]

new_shap_values = []
for values in shap_values_categorical.values:
    values_split = np.split(values , np.cumsum(n_categories)) #split shap values into a list for each feature
    values_sum = [sum(l) for l in values_split] #sum values within each list
    new_shap_values.append(values_sum)

# Replace shap values for categorical variables
new_shap_values_categorical = np.array(new_shap_values)

categorical_variables_name = ["qualifications", "ownORrent_accomodation_lived",
                              "current_employment_status",
                              "weekly_usage_of_mobile_phone_in_last_3_months",
                              "leisure_social_activities", "Null"]

new_shap_values_categorical_name = pd.DataFrame(new_shap_values_categorical, columns = categorical_variables_name)

categorical_shap_values_name = new_shap_values_categorical_name.drop('Null', axis = 1)

# Concatenate shap values for categorial and continue variables
continue_shap_value = shap_values_tree_name.drop(feature_names, axis = 1)

Final_shap_value = np.concatenate((continue_shap_value, categorical_shap_values_name), axis = 1)
columns_for_final_shap_value = ["Age", "Standing height", "Right hand grip strength",
                                "Number in household", "Cystatin C",
                                "Glycated haemoglobin (HbA1c)", "IGF1",
                                "Systolic blood pressure (Automated reading)",
                                "Weighted PGS", "Sex", "Qualifications",
                                "Own or rent accommodation lived",
                                "Current employment status",
                                "Weekly usage of mobile phone in last 3 months",
                                "Leisure/social activities"]

Final_shap_values_name = pd.DataFrame(Final_shap_value, columns = columns_for_final_shap_value)

rus_2 = RandomUnderSampler(sampling_strategy = 0.50, random_state = 546)
X_res_2, y_res_2 = rus.fit_resample(df_no_Pheno, y)

# Convert pandas df to np 
prova_1 = Final_shap_values_name.to_numpy()
prova_2 = X_res_2.to_numpy()
final_shap_values_tree = pd.DataFrame(prova_1, columns=columns_for_final_shap_value)
final_db_plot = pd.DataFrame(prova_2, columns=columns_for_final_shap_value)

# Summary_plot and dependence plot
fig = plt.subplots(dpi = 1000)
plt.title("MLP", y = 1.02, x = 0.42)
shap.summary_plot(prova_1, prova_2, feature_names = columns_for_final_shap_value)
shap.summary_plot(prova_1, prova_2, feature_names = columns_for_final_shap_value, plot_type = "bar")

for ind in final_shap_values_tree.columns:
        shap.dependence_plot("age", prova_1, prova_2, feature_names=columns_for_final_shap_value, interaction_index = (ind))
        
for ind in final_shap_values_tree.columns:
        shap.dependence_plot("weighted", prova_1, prova_2, feature_names=columns_for_final_shap_value, interaction_index = (ind))

for var in final_shap_values_tree.columns:
    shap.dependence_plot(var, prova_1, prova_2, feature_names = columns_for_final_shap_value)
    
    
