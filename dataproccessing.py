#!/usr/bin/env python3
"""dataproccessing.py: *** Note use of parameter testing bools"""

__author__ = "Reina Hastings, Eugenio Casta"
__email__ = "reinahastings13@gmail.com"

import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import re
import pickle
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, multilabel_confusion_matrix, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier

# Specify output file for debugging purposes.
log_file = 'nn_parameter_testing_log.txt'

# Open the file and redirect standard output to it.
with open(log_file, 'a') as f:
    sys.stdout = f
    
    # Print time stamp for new log entry.
    current_time = datetime.now()
    print('Timestamp: ', current_time)

    def row_validation(row):
        for cell in row:
            # Check for NaN, None, or blank strings and filter them out
            if cell is None or cell == '' or (isinstance(cell, float) and np.isnan(cell)):
                return False
        return True

    def classification(imported_data, model, multi=False):
        # Returns Pandas df containing multi-class binary matrix of the 'classification' column in addition to columns of interest: ['structureId',
        #    'transport', 'adhesion', 'other_binding', 'transferase', 'metal', 'viral', 'inhibitor', 'membrane', 'oxidoreductase', 'RNA', 'structural',
        #    'lyase', 'DNA_RNA_binding', 'transcription', 'signal', 'immune', 'chaperone', 'isomerase', 'hydrolase', 'protein_binding', 'DNA',
        #    'regulation', 'genomics', 'ligase', 'macromoleculeType', 'residueCount', 'resolution', 'structureMolecularWeight', 'densityMatthews',
        #    'densityPercentSol', 'phValue']
        # Removes all rows with missing values or rows with classifications that don't correspond to the values in the new classification matrix.
        # protein_data = create_multiclass_matrix(imported_data)
        
        # Use pickled data frame for testing efficiency purposes.
        # protein_data.to_pickle('protein_data.pkl')
        protein_data = pd.read_pickle('protein_data.pkl')
        print(f'Length of protein_data: {len(protein_data)}')
        
        # # Convert sample_class from pd to np array.
        # sample_class = sample_class.to_numpy()
        
        # dataset = np.array([x0,x1,x2,x3,x4,x5,sample_class], dtype=object)
        # dataset = dataset.T
        
        # # Remove rows that fail row_validation check (contains NaN, None, or '').
        # dataset = np.array([row for row in dataset if row_validation(row)], dtype=object)
        
        # encoder = LabelEncoder()
        # for i in range(dataset.shape[1]):
        #     if dataset[:, i].dtype == object:  # Check if column has strings
        #         dataset[:, i] = encoder.fit_transform(dataset[:, i])
        
        # One-hot encoding of 'macromoleculeType' variable since it is nominal.
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(protein_data[['macromoleculeType']])
        
        # Turn encoded from NumPy array to Pandas array.
        feature_names = encoder.get_feature_names_out(['macromoleculeType'])
        encoded = pd.DataFrame(encoded, columns=feature_names).astype(int)
        
        # # Check if df is a DataFrame
        # is_dataframe = isinstance(encoded, pd.DataFrame)
        # print(is_dataframe)  # This will print: True
        
        # Optional: Save encoded data to a csv file to visually inspect.
        encoded.to_csv('encoded.csv', index=True)
        
        # Drop 'macromoleculeType' column from protein_data.
        protein_data = protein_data.drop('macromoleculeType', axis=1)
        
        # Add encoded data frame containing the one-hot encoded 'macromoleculeType' data to protein_data.
        # *** Note to self: If I have issues reassigning the structureIds back onto the df, check this line since the indexing could have been tampered with. ***
        # print(protein_data.index)
        # print(encoded.index)
        
        # Debugging: Check if protein_data and encoded indices are aligned for concatenation.
        # if protein_data.index.equals(encoded.index):
        #     print("Indices are aligned!")
        # else:
        #     print("Indices are not aligned!")   
        # print(protein_data.dtypes)
        # print(encoded.dtypes)
        
        protein_data = pd.concat([protein_data,encoded], axis=1)
        
        # Optional: Save protein_data to a csv file to visually inspect.
        protein_data.to_csv('protein_data.csv', index=True)
        
        # print('Protein data: ')
        # print(protein_data.shape)
        
        # dataset = dataset.astype(np.float32)
        
        # Keep copy of structureId column.
        structure_IDs = protein_data['structureId']
        
        # Drop the 'structureId' column from data set before training.
        protein_data = protein_data.drop(columns=['structureId'])
        
        # label_matrix includes the first 24 columns which make up the classifier binary matrix.
        label_matrix = protein_data.iloc[:, 0:24]
        # print(f'Length of label_matrix: {len(label_matrix)}')
        
        # features includes the rest of the data frame after the first 24 columns.
        features = protein_data.iloc[:, 24:]
        # print(f'Length of features: {len(features)}')
        
        # Optional: Save features and label_matrix to a csv files to visually inspect.
        label_matrix.to_csv('label_matrix.csv', index=True)
        features.to_csv('features.csv', index=True)
        
        if multi==True:
            # First Split: Train (80%) and validation + test (20%) sets.
            X_train, X_test_val, y_train, y_test_val = train_test_split(features, label_matrix, test_size=0.2, random_state=42)
            
            # Optional: Check shape of train, validation + test sets.
            # print('Shape of train/validation+test: ')
            # print(X_train.shape)
            # print(y_train.shape)
            # print(X_test_val.shape)
            # print(y_test_val.shape)
            
            # Second Split: Train (50%) and validation (50%) sets.
            X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
            
            # Total data set split into train (80%), validation (10%), and test (10%).
            
            # Optional: Check shape of train, validation, and test sets.
            # print('Shape of test/train/val: ')
            # print(X_test.shape)
            # print(y_test.shape)
            # print(X_train.shape)
            # print(y_train.shape)
            # print(X_val.shape)
            # print(y_val.shape)
            
            # print('\nFeature names and label names: ')
            # print(feature_names)
            # print(label_names)
            
            # list = [X_test, y_test, X_train, y_train, X_val, y_val]
            # for df in list:
            #     if isinstance(df, np.ndarray):
            #         print(f"{df} is a NumPy array.")
                
            #     if isinstance(df, pd.DataFrame):
            #         print(f"{df} is a Pandas DataFrame.")
            
            # # Explore data distribution and statistics.
            # num_labels = label_matrix.iloc[:, :6] # Numerical features only, one-hot encoded feature included separately in encoded.
            # print(num_labels)
            # generate_summary_statistics(features, num_labels, encoded)
            
            # visualize_data(protein_data)
            
            # Retrieve feature_names and label_names to identify failure cases later.
            feature_names = features.columns.tolist()
            label_names = label_matrix.columns.tolist()
            
            # Convert Pandas data frames to NumPy arrays.
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
            
            if model == 'rf':
                random_forest(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names, run_param_test=False)
            elif model == 'dt':
                decision_tree(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names)
            elif model == 'nn':
                neural_network(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names)
        
        else:
            label_matrix, features = remove_multi_label_samples(label_matrix, features)
            
            print("Filtered Labels:")
            print(label_matrix)
            
            print("Filtered Features:")
            print(features)
            
            # First Split: Train (80%) and validation + test (20%) sets.
            X_train, X_test_val, y_train, y_test_val = train_test_split(features, label_matrix, test_size=0.2, random_state=42)
            
            # Second Split: Train (50%) and validation (50%) sets.
            X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
            
            # Retrieve feature_names and label_names to identify failure cases later.
            feature_names = features.columns.tolist()
            label_names = label_matrix.columns.tolist()
            
            # Convert Pandas data frames to NumPy arrays.
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
            
            # if model == 'rf':
            #     random_forest(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names, run_param_test=False)
            # elif model == 'dt':
            #     decision_tree(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names)
            # elif model == 'nn':
            #     neural_network(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names)

    def random_forest(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names, run_param_test):
        """
        Train and evaluate a Random Forest model with multi-label outputs.

        Args:
            - X_train, X_test, X_val : pd.DataFrame
                Features for training, testing, and validation, respectively.

            - y_train, y_test, y_val : pd.DataFrame
                Labels for training, testing, and validation, respectively.

            - feature_names : list of str
                Names of the feature columns.

            - label_names : list of str
                Names of the label columns.

            - run_param_test : bool, default=False
                If True, runs parameter testing module to determine the best parameters for the model.
                If False, uses predefined default parameters.

        Returns:
            - None
                Prints evaluation metrics and failure cases for the model on validation and test sets.
        """
        
        # Initialize base model.
        base_rf = RandomForestClassifier(random_state=42)
        
        # Predefined parameters for testing.
        default_params = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }
        
        if run_param_test:
            print("Running parameter testing...")
            best_params = rf_parameter_testing(X_train, y_train)  # Trigger the parameter testing module.
        else:
            print("Using default parameters...")
            best_params = default_params
            
        print(f"Selected Parameters: {best_params}")
        
        # Initialize the Random Forest model with the chosen parameters.
        base_rf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=42,
        )
        
        # Wrap model with MultiOutputClassifier.
        multi_target_rf = MultiOutputClassifier(base_rf, n_jobs=-1)
        
        # Train the model on the training set.
        print("Training the model...")
        multi_target_rf.fit(X_train, y_train)
        
        # Evaluate on validation set.
        y_pred_val = multi_target_rf.predict(X_val)
        
        # Generate confusion matrix and scoring metrics for multi-output random forest classifier applied to validation set.    
        c_matrix(y_val, y_pred_val, label='Random Forest Multi-Output Classifier on Validation Set')
        
        # log_failure_cases
        log_failure_cases(X_val, y_val, y_pred_val, label_names, feature_names, label='Random Forest Multi-Output Classifier Validation Set Failure Cases')
        
        # Evaluate on test set.
        y_pred_test = multi_target_rf.predict(X_test)
        
        # Generate confusion matrix and scoring metrics for multi-output random forest classifier applied to test set.    
        c_matrix(y_test, y_pred_test, label='Random Forest Multi-Output Classifier on Test Set')
        
        # log_failure_cases
        log_failure_cases(X_test, y_test, y_pred_test, label_names, feature_names, label='Random Forest Multi-Output Classifier Test Set Failure Cases')
        
        print('Timestamp: ', current_time)
        print('Finished testing Random Forest model.')

    def decision_tree(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names):
        """
        Train and evaluate a multi-label Decision Tree Classifier hyperparameter tuning.
        
        Args:
            - features : np.array or pd.DataFrame
                Input features for training, validation, and testing.
            
            - labels : np.array or pd.DataFrame
                Multi-label binary matrix as the target. Each row corresponds to a sample, and each 
                column represents a label (1 for presence, 0 for absence).
        
        Returns:
            - None
                This function does not return a value but:
                - Trains a Decision Tree Classifier on the training set.
                - Evaluates the model on the validation set and prints metrics including accuracy, 
                precision, recall, and F1 score.
                - Optionally evaluates the model on the test set and prints similar metrics.
                
        Notes:
        ------
        - The training set is used to fit the Decision Tree Classifier.
        - The validation set is used for hyperparameter tuning and model evaluation.
        - The test set is optionally used for final model evaluation.
        - The `c_matrix` function is used to compute and display evaluation metrics for each set.
        
        Example Usage:
        --------------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> features, labels = some_feature_matrix, some_label_matrix
        >>> train_evaluate_decision_tree(features, labels)
        """
        # Initialize base Decision Tree model.
        base_tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
        
        # Wrap the model with MultiOutputClassifier for multi-label support.
        multi_tree = MultiOutputClassifier(base_tree)
        
        # Define the parameter grid for GridSearchCV.
        param_grid = {
            'estimator__criterion': ['gini', 'entropy'],
            'estimator__max_depth': [None, 5, 10, 15],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 5]
        }
        
        # Initialize GridSearchCV.
        grid_search = GridSearchCV(
            estimator=multi_tree,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation.
            scoring='accuracy',  # Use accuracy as the evaluation metric.
            verbose=2,
            n_jobs=-1  # Use all available processors.
        )
        
        # Fit GridSearchCV on the training data.
        grid_search.fit(X_train, y_train)
        
        # Train the model.
        multi_tree.fit(X_train, y_train)
        
        # Retrieve the best parameters and best model.
        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
        
        # Retrieve all tested parameter combinations and their metrics.
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Extract relevant columns.
        metrics_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
        
        # Print all parameter combinations with their metrics.
        print("\nMetrics for All Parameter Combinations:")
        print(metrics_df)
        
        # Print metrics for the best parameter combination.
        best_params_metrics = metrics_df[metrics_df['rank_test_score'] == 1]
        print("\nMetrics for Best Parameters:")
        print(best_params_metrics)
        
        # Evaluate on validation set.
        y_pred_val = best_model.predict(X_val)
        
        # Generate confusion matrix and scoring metrics for multi-output random forest classifier applied to validation set.    
        c_matrix(y_val, y_pred_val, label='Decision Tree Multi-Output Classifier on Validation Set')
        
        # log_failure_cases
        log_failure_cases(X_val, y_val, y_pred_val, label_names, feature_names, label='Decision Tree Multi-Output Classifier Validation Set Failure Cases')
        
        # Evaluate on test set.
        y_pred_test = best_model.predict(X_test)
        
        # Generate confusion matrix and scoring metrics for multi-output random forest classifier applied to test set.    
        c_matrix(y_test, y_pred_test, label='Decision Tree Multi-Output Classifier on Test Set')
        
        # log_failure_cases
        log_failure_cases(X_test, y_test, y_pred_test, label_names, feature_names, label='Decision Tree Multi-Output Classifier Test Set Failure Cases')
        
        print('Timestamp: ', current_time)
        print('Finished testing Decision Tree model.')

    def create_neural_model(learning_rate=1e-3, num_neurons=32, activation='relu', optimizer='adam'):
        """
        Create a neural network model with the specified hyperparameters.
        
        Args:
            - learning_rate : float
                The learning rate for the optimizer.
            - num_neurons : int
                The number of neurons in the hidden layers.
            - activation : str
                The activation function to use in the hidden layers.
            - optimizer : str
                The optimizer to use ('adam' or 'sgd').
        
        Returns:
            - model : Keras model
                A compiled Keras model ready for training.
        """
        y_train_shape = (74104, 24)
        X_train_shape = (74104, 19)
        
        # Define optimizers.
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate)
        }
        opt = optimizers[optimizer]
        
        # Build the model.
        model = Sequential([
            Input(shape=(X_train_shape[1],)),
            Dense(num_neurons, activation=activation),
            Dense(num_neurons * 2, activation=activation),
            Dense(y_train_shape[1], activation='sigmoid')  # Output layer.
        ])
        
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def neural_network(X_train, X_test, X_val, y_train, y_test, y_val, feature_names, label_names):
        """
        Train and evaluate a neural network for multi-label binary classification with hyperparameter tuning.
        
        Args:
            - X_train, X_test, X_val : np.array or pd.DataFrame
                Input feature matrices for training, testing, and validation sets.
            - y_train, y_test, y_val : np.array or pd.DataFrame
                Multi-label binary target matrices for training, testing, and validation sets.
        
        Returns:
            - None
                Prints evaluation metrics for validation and test sets.
        """
        
        # Wrap the Keras model in a scikit-learn wrapper.
        model = KerasClassifier(build_fn=create_neural_model, verbose=0)

        # Define the parameter grid for the grid search.
        # 'model__' prefix passes arguments to create_neural_model.
        # 'fit__' passes arguments to .fit() method of KerasClassifier.
        param_grid = {
            'model__learning_rate': [1e-3, 1e-4],
            'model__num_neurons': [32, 64],
            'model__activation': ['relu', 'tanh'],
            'model__optimizer': ['adam', 'sgd'],
            'fit__epochs': [10, 20],
            'fit__batch_size': [16, 32, 64],
        }
        
        # Perform GridSearchCV.
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Retrieve and display metrics for all tested parameter combinations.
        results = pd.DataFrame(grid_search.cv_results_)
        print("\nGrid Search Results:\n")
        for index, row in results.iterrows():
            print(f"Params: {row['params']}, Mean Accuracy: {row['mean_test_score']:.4f}, Std Accuracy: {row['std_test_score']:.4f}")
            
        # Best parameters and their metrics.
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        print("\nBest Parameters:", best_params)
        
        # Evaluate the best model on validation set.
        y_pred_val_prob = best_model.predict(X_val)
        y_pred_val = (y_pred_val_prob > 0.5).astype(int)
        
        # Generate confusion matrix and scoring metrics for multi-output neural network applied to validation set.    
        c_matrix(y_val, y_pred_val, label='Neural Network Multi-Output Classifier on Validation Set')
        
        # log_failure_cases
        log_failure_cases(X_val, y_val, y_pred_val, label_names, feature_names, label='Neural Network Multi-Output Classifier Validation Set Failure Cases')
        
        # Evaluate the best model on test set.
        y_pred_test_prob = best_model.predict(X_test)
        y_pred_test = (y_pred_test_prob > 0.5).astype(int)
        
        # Generate confusion matrix and scoring metrics for multi-output neural network applied to test set.    
        c_matrix(y_test, y_pred_test, label='Neural Network Multi-Output Classifier on Test Set')
        
        # log_failure_cases
        log_failure_cases(X_test, y_test, y_pred_test, label_names, feature_names, label='Neural Network Multi-Output Classifier Test Set Failure Cases')
        
        print('Timestamp: ', current_time)
        print('Finished testing Neural Network model.')

    def c_matrix(y_true, y_pred, label):
        """
        Evaluates the performance of a multilabel classification model using various metrics.
        
        Args:
            - y_true : array-like of shape (n_samples, n_classes)
                Ground truth target values in a binary multilabel format. Each row corresponds
                to a sample, and each column corresponds to a label (1 for presence, 0 for absence).
        
            - y_pred : array-like of shape (n_samples, n_classes)
                Predicted target values in a binary multilabel format. Must match the shape of `y_true`.
        
            - label : str, default="Model Evaluation"
                A descriptive label to identify the output of the evaluation (e.g., model name, dataset).
        
        Returns:
            - None
                This function does not return a value but prints:
                - A label identifying the evaluation context.
                - Confusion matrices for each label.
                - Evaluation metrics including accuracy, precision, recall, and F1 score.
                
        Notes:
        ------
        - The function uses the `multilabel_confusion_matrix` from `sklearn` to compute confusion matrices 
        for each label independently.
        - The metrics use `average='weighted'` to handle imbalanced labels effectively.
        - The `zero_division=1` parameter ensures that precision and recall calculations do not fail
        due to division by zero.
        
        Example Usage:
        --------------
        >>> from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        >>> c_matrix(y_true, y_pred, label="Random Forest Multi-Output Classifier")
        """
        
        print(f"Evaluation for: {label}")
        print("=" * (14 + len(label)))
        
        # Compute multilabel confusion matrix.
        ml_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        
        # Print confusion matrix for each label.
        for i, cm in enumerate(ml_confusion_matrix):
            print(f'\\Confusion matrix for label {i}: ')
            print(cm)
        
        # Compute and print evaluation metrics.
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted',zero_division=1)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def create_multiclass_matrix(imported_data='pdb_data_no_dups.csv', matrix_creation_log='True'):
        """
        Creates a multi-class binary matrix based on the classification data from a protein data set.
        
        Args:
            - imported_data : str or pd.DataFrame, default='pdb_data_no_dups.csv'
                The input data set containing protein classification data. If a string is passed, 
                it should be the file path to a CSV file. If a DataFrame is passed, it is used directly.
            
            - matrix_creation_log : str, default='True'
                A flag to enable or disable logging of the matrix creation process. 
                - If 'True': Logs detailed information about row processing, including keyword matches.
                - If 'False': Suppresses logging to improve performance.
        
        Returns:
            - pd.DataFrame
                A DataFrame containing the binary multi-class target matrix:
                - Columns represent the subclasses (e.g., 'DNA', 'RNA', 'protein_binding').
                - Rows represent samples, identified by their `structureId`.
                - Each cell is 0 or 1, indicating the presence or absence of a subclass for a sample.
                - Rows with no subclass matches (all zeros) are removed.
            
            Notes:
            -----
            - The method identifies subclasses based on keywords in the 'classification' column.
            - Keywords are mapped to subclasses using the `keyword_to_subclass` dictionary.
            - Special handling is applied to certain keywords (e.g., 'binding', 'dna', 'rna') to ensure logical classification.
            - If `matrix_creation_log` is 'True', detailed processing logs are saved to a file named `matrix_creation_log.txt`.
            
        Example Usage:
        --------------
        >>> create_multiclass_matrix('protein_data.csv', matrix_creation_log='False')
        """
        
        # Select columns of interest.
        protein_data = imported_data[['structureId','classification', 'macromoleculeType', 'residueCount', 'resolution',
                                    'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue']]
        
        # Remove rows with missing values.
        protein_data = protein_data.dropna()
        
        # Keywords selected based on number of occurrence. To be considered a keyword, the word(s) must appear more than 1,000 times in the data set.
        keywords = ['structural','lyase','genomics','signal','transport','metal','membrane','isomerase','oxidoreductase','ligase','protein binding',
                    'protein-binding','adhesion','chaperone','RNA','DNA','binding','rna binding','rna-binding','viru',
                    'transferase','hydrolase','inhibitor','transcription','immune','genomics','regulator','regulation','viral','dna binding',
                    'dna-binding']
        
        subclasses = ['hydrolase','transferase','oxidoreductase','DNA_RNA_binding','protein_binding','other_binding','inhibitor','transport',
                    'DNA','RNA','transcription','immune','structural','isomerase','signal','ligase','viral','genomics','metal','membrane','chaperone',
                    'adhesion','regulation']
        
        # Keyword to Subclass Mapping
        keyword_to_subclass = {
            'structural': 'structural',
            'lyase': 'lyase',
            'genomics': 'genomics',
            'signal': 'signal',
            'transport': 'transport',
            'metal': 'metal',
            'membrane': 'membrane',
            'isomerase': 'isomerase',
            'oxidoreductase': 'oxidoreductase',
            'ligase': 'ligase',
            'protein binding': 'protein_binding',
            'protein-binding': 'protein_binding',
            'adhesion': 'adhesion',
            'chaperone': 'chaperone',
            'rna': 'RNA',
            'dna': 'DNA',
            'rna binding': 'DNA_RNA_binding',
            'rna-binding': 'DNA_RNA_binding',
            'dna binding': 'DNA_RNA_binding',
            'dna-binding': 'DNA_RNA_binding',
            'binding': 'other_binding',
            'viru': 'viral',
            'viral': 'viral',
            'transferase': 'transferase',
            'hydrolase': 'hydrolase',
            'inhibitor': 'inhibitor',
            'transcription': 'transcription',
            'immune': 'immune',
            'genomics': 'genomics',
            'regulator': 'regulation',
            'regulation': 'regulation',
        }
        
        # Extract unique subclass names from the mapping.
        subclasses = set(keyword_to_subclass.values())
        
        # Initialize the target matrix with all zeros.
        target_matrix = pd.DataFrame(0, index=protein_data.index, columns=list(subclasses))
        
        # Safeguard structureId column.
        target_matrix.insert(0, 'structureId', protein_data['structureId'])
        
        # Copy structureID into the target matrix.
        target_matrix['structureId'] = protein_data['structureId']
        
        # Ensure the indices of protein_data and target_matrix are aligned before populating the matrix.
        protein_data.reset_index(drop=True, inplace=True)
        target_matrix.reset_index(drop=True, inplace=True)
        
        # Conditional logging for matrix creation.
        if matrix_creation_log == 'True':
            
            # Specify output file for matrix creation log.
            matrix_creation_file = 'matrix_creation_log.txt'
            
            # Open the file and redirect standard output to it.
            with open(matrix_creation_file, 'w') as f:
                sys.stdout = f
                
                # Populate the target matrix.
                for i, text in enumerate(protein_data['classification']):
                    structure_id = protein_data.iloc[i]['structureId']  # Access structureId for the current row
                    text_lower = text.lower()  # Convert to lowercase for case-insensitive matching
                    
                    # matrix_creation: Start processing a new row
                    print(f"\nProcessing row {i}, StructureID: {structure_id}")
                    print(f"Original text: {text_lower}")
                    
                    # Special case: Initialize flags for 'DNA_RNA_binding' and 'protein_binding'.
                    triggers_DNA_RNA_binding = False
                    triggers_protein_binding = False
                    
                    # Check for each keyword in the text
                    for keyword, subclass in keyword_to_subclass.items():
                        if re.search(fr'{keyword}', text_lower): # Match keyword in text.
                            print(f"Match found: '{keyword}' -> Subclass: '{subclass}'")
                            if subclass == 'DNA_RNA_binding':
                                triggers_DNA_RNA_binding = True
                                target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                                print(f"    Marking 'DNA_RNA_binding' (triggers_DNA_RNA_binding=True)")
                            elif subclass == 'protein_binding':
                                triggers_protein_binding = True
                                target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                                print(f"    Marking 'protein_binding' (triggers_protein_binding=True)")
                            elif subclass != 'other_binding' and subclass != 'DNA': # Handles general subclasses.
                                target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                                print(f"    Marking subclass: '{subclass}'")
                    
                    # Special case: Handle 'other_binding' class ('binding' keyword trigger that is not mapped to DNA_RNA_binding or protein_binding).
                    if re.search(r'binding',text_lower) and not triggers_DNA_RNA_binding and not triggers_protein_binding:
                        target_matrix.iloc[i, target_matrix.columns.get_loc('other_binding')] = 1
                        print(f"  Special case: Marking 'other_binding'")
                    
                    # Special case: Handle 'dna' keyword for 'DNA' class only if not already 'DNA_RNA_binding' class.
                    if re.search(r'dna',text_lower) and not triggers_DNA_RNA_binding:
                        target_matrix.iloc[i, target_matrix.columns.get_loc('DNA')] = 1
                        print(f"  Special case: Marking 'DNA' (not triggered by 'DNA_RNA_binding')")
                    
                    # Special case: Handle 'rna' keyword for 'RNA' class only if not already 'DNA_RNA_binding' class.
                    if re.search(r'rna',text_lower) and not triggers_DNA_RNA_binding:
                        target_matrix.iloc[i, target_matrix.columns.get_loc('RNA')] = 1
                        print(f"  Special case: Marking 'RNA' (not triggered by 'DNA_RNA_binding')")
                        
            # Reset standard output back to the console.
            sys.stdout = sys.__stdout__
            
            # Inform user of the matrix_creation log file.
            print(f"matrix_creation logs have been written to '{matrix_creation_file}'.")
        
        else:
            # Populate the target matrix.
            for i, text in enumerate(protein_data['classification']):
                structure_id = protein_data.iloc[i]['structureId']  # Access structureId for the current row.
                text_lower = text.lower()  # Convert to lowercase for case-insensitive matching.
                
                # matrix_creation: Start processing a new row.
                print(f"\nProcessing row {i}, StructureID: {structure_id}")
                print(f"Original text: {text_lower}")
                
                # Special case: Initialize flags for 'DNA_RNA_binding' and 'protein_binding'.
                triggers_DNA_RNA_binding = False
                triggers_protein_binding = False
                
                # Check for each keyword in the text.
                for keyword, subclass in keyword_to_subclass.items():
                    if re.search(fr'{keyword}', text_lower): # Match keyword in text.
                        print(f"Match found: '{keyword}' -> Subclass: '{subclass}'")
                        
                        # Check for special case subclasses: DNA_RNA_binding, protein_binding, other_binding, DNA, RNA
                        if subclass == 'DNA_RNA_binding':
                            triggers_DNA_RNA_binding = True
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                            
                        elif subclass == 'protein_binding':
                            triggers_protein_binding = True
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                            
                        elif subclass != 'other_binding' and subclass != 'DNA': # Handles general subclasses.
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                
                # Special case: Handle 'other_binding' class ('binding' keyword trigger that is not mapped to DNA_RNA_binding or protein_binding).
                # Prevents 'other_binding' being triggered when 'DNA_RNA_binding' or 'protein_binding' is triggered.
                if re.search(r'binding',text_lower) and not triggers_DNA_RNA_binding and not triggers_protein_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('other_binding')] = 1
                
                # Special case: Handle 'dna' keyword for 'DNA' class only if not already 'DNA_RNA_binding' class.
                # Prevents 'dna' being triggered when 'DNA_RNA_binding' is triggered.
                if re.search(r'dna',text_lower) and not triggers_DNA_RNA_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('DNA')] = 1
                
                # Special case: Handle 'rna' keyword for 'RNA' class only if not already 'DNA_RNA_binding' class.
                # Prevents 'rna' being triggered when 'DNA_RNA_binding' is triggered.
                if re.search(r'rna',text_lower) and not triggers_DNA_RNA_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('RNA')] = 1
        
        # Remove rows that are all zeros (excluding 'structureId').
        non_zero_matrix = target_matrix.loc[(target_matrix.iloc[:, 1:] != 0).any(axis=1)]
        
        # Columns to add back to the matrix to return a complete data set.
        columns_to_add = [
            'macromoleculeType', 'residueCount', 'resolution',
            'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue'
        ]
        
        # Merge the columns based on `structureId`.
        if 'structureId' in non_zero_matrix.columns and 'structureId' in protein_data.columns:
            matrix_protein_data = non_zero_matrix.merge(
                protein_data[['structureId'] + columns_to_add],  # Select columns to merge.
                on='structureId',  # Merge on the structureId column.
                how='left'  # Keep all rows in non_zero_matrix.
            )
        else:
            print("Error: 'structureId' must exist in both non_zero_matrix and data set.")
        
        # Display the updated matrix_data_set structure.
        print(matrix_data_set.head())
        
        # Save matrix_data_set to a csv file.
        matrix_data_set.to_csv('matrix_data_set.csv', index=True)
        
        return (matrix_protein_data)

    def rf_parameter_testing(X_train, y_train):
        """
        Perform hyperparameter for Random Forest algorithm tuning using GridSearchCV.

        Args:
            - X_train : np.array
                Training features.

            - y_train : np.array
                Training labels.

        Returns:
            - best_params : dict
                Best parameters found by the grid search.
        """
        # Define parameter grid for model tuning.
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],  # Number of trees - higher generally improves performance but increases computation time.
            'estimator__max_depth': [10, 20, None],     # Max depth of each tree - shallower trees are faster and less prone to overfitting but deeper trees can capture more complex patterns.
            'estimator__min_samples_split': [2, 5, 10], # Minimum samples required to split a node - larger values reduce the likelihood of overfitting but can constrain the splits.
            'estimator__min_samples_leaf': [1, 2, 4],   # Minimum samples required at each leaf node - larger values prevent the model from learning overly specific patterns -> overfitting. But can lead to underfitting.
        }

        # Initialize the base model.
        base_rf = RandomForestClassifier(random_state=42)

        # Wrap the base model for multi-label classification.
        multi_target_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

        # Perform grid search.
        print("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            multi_target_rf,
            param_grid,
            scoring=make_scorer(accuracy_score, greater_is_better=True),
            cv=3,
            verbose=2,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Access results.
        results = grid_search.cv_results_

        # Print parameters and accuracy for each combination.
        print("\nGrid Search Results:")
        for i in range(len(results["params"])):
            params = results["params"][i]
            mean_accuracy = results["mean_test_score"][i]  # Mean accuracy across all CV folds.
            std_accuracy = results["std_test_score"][i]    # Standard deviation across folds.
            print(f"Parameters: {params}")
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"Standard Deviation: {std_accuracy:.4f}")
            print("-" * 40)

        # Print best parameters found and their metrics.
        print("\nBest Parameters and Metrics:")
        best_index = grid_search.best_index_  # Index of the best parameters.
        best_params = grid_search.best_params_
        best_mean_accuracy = results["mean_test_score"][best_index]
        best_std_accuracy = results["std_test_score"][best_index]
        
        # **** Current error: best_params has trouble accessing n_estimators but I don't want to rerun this yet because it takes like a fucking hour. *****
        print(best_params)
        
        print(f"Best Parameters: {best_params}")
        print(f"Mean Accuracy: {best_mean_accuracy:.4f}")
        print(f"Standard Deviation: {best_std_accuracy:.4f}")

        # Return the best parameters.
        return best_params

    def log_failure_cases(X, y_true, y_pred, label_names, feature_names, label="Model"):
        """
        Prints failure cases where the predicted labels do not match the true labels.

        Args:
            - X : np.array of shape (n_samples, n_features)
                Input features corresponding to the samples.

            - y_true : np.array of shape (n_samples, n_classes)
                Ground truth target values in a binary multilabel format.

            - y_pred : np.array of shape (n_samples, n_classes)
                Predicted target values in a binary multilabel format.

            - label_names : list of str
                Names of the labels corresponding to the target columns.

            - feature_names : list of str
                Names of the features corresponding to the input columns.

            - label : str, default="Model"
                A descriptive label to identify the model or dataset being evaluated.

        Returns:
            - None
                Prints detailed information about the failure cases to the console.

        Notes:
        ------
        - The function identifies failure cases where the true labels do not match the predicted labels.
        - Prints the feature values, true labels, predicted labels, and feature/label names for context.
        - A random selection of 10 failure cases is printed.

        Example Usage:
        --------------
        >>> log_failure_cases(X_test, y_test, y_pred, label_names, feature_names, label="Random Forest Classifier")
        """

        # Ensure inputs are numpy arrays.
        if not (isinstance(X, np.ndarray) and isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
            raise ValueError("Inputs X, y_true, and y_pred must be numpy arrays.")

        print(f"Shape of X: {X.shape}")
        print(f"Shape of y_true: {y_true.shape}")
        print(f"Shape of y_pred: {y_pred.shape}")

        print(f"Feature dtypes: {X.dtype}")
        print(f"Label dtypes (y_true): {y_true.dtype}")
        print(f"Label dtypes (y_pred): {y_pred.dtype}")

        # Collect all failure cases.
        failure_cases = []

        for i, (x, true, pred) in enumerate(zip(X, y_true, y_pred)):
            if not np.array_equal(true, pred):  # Check if true and predicted labels are different.
                failure_cases.append((i, x, true, pred))

        # Randomly select 10 failure cases or fewer if less are available.
        selected_cases = np.random.choice(len(failure_cases), min(10, len(failure_cases)), replace=False)

        print(f"\nFailure Cases for {label}:")
        print("=" * 80)

        for idx in selected_cases:
            i, x, true, pred = failure_cases[idx]
            print(f"Sample {i}:")
            print(f"  Features:")
            for feature_name, feature_value in zip(feature_names, x):
                print(f"    {feature_name}: {feature_value}")

            print("  True Labels:")
            for label_name, label_value in zip(label_names, true):
                print(f"    {label_name}: {label_value}")

            print("  Predicted Labels:")
            for label_name, label_value in zip(label_names, pred):
                print(f"    {label_name}: {label_value}")

            print("-" * 80)

        print(f"End of Failure Cases for {label}")
        print("=" * 80)

    def generate_summary_statistics(features_df, labels_df, categorical_df):
        """
        Generates summary statistics and visualizations for features, labels, and one-hot encoded categorical features DataFrames.

        Args:
            - features_df (pd.DataFrame): DataFrame containing the numeric features.
            - labels_df (pd.DataFrame): DataFrame containing the binary multilabel target matrix.
            - categorical_df (pd.DataFrame): DataFrame containing the one-hot encoded categorical features.
            - output_dir (str, optional): Directory to save plots. Defaults to the current directory.

        Returns:
            - None
        """

        # Summary Statistics for Numeric Features
        print("Summary Statistics for Numeric Features:")
        print(features_df.describe())

        # Missing Values in Numeric Features
        print("\nMissing Values in Numeric Features:")
        print(features_df.isnull().sum())

        # Histograms for Numeric Features
        print("\nGenerating histograms for numeric features...")
        features_df.hist(bins=20, figsize=(10, 8))
        plt.suptitle("Histograms of Numeric Features")
        plt.savefig("features_histograms.png")
        plt.show()

        # Boxplots for Numeric Features
        print("\nGenerating boxplots for numeric features...")
        features_df.boxplot(figsize=(10, 8))
        plt.title("Boxplots of Numeric Features")
        plt.savefig("features_boxplots.png")
        plt.show()

        # KDE Plots for Numeric Features
        print("\nGenerating KDE plots for numeric features...")
        for column in features_df.select_dtypes(include=["float64", "int64"]):
            features_df[column].plot(kind="kde")
            plt.title(f"KDE Plot for {column}")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.savefig("{column}_kde.png")
            plt.show()

        # Correlation Matrix for Numeric Features
        print("\nGenerating correlation matrix for numeric features...")
        correlation_matrix = features_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Numeric Features")
        plt.savefig("features_correlation_matrix.png")
        plt.show()

        # Summary Statistics for One-Hot Encoded Categorical Features
        print("\nSummary Statistics for One-Hot Encoded Categorical Features:")
        print(categorical_df.describe())

        # Missing Values in Categorical Features
        print("\nMissing Values in Categorical Features:")
        print(categorical_df.isnull().sum())

        # Distribution of Each Categorical Feature
        print("\nGenerating bar plots for one-hot encoded categorical features...")
        for column in categorical_df.columns:
            categorical_df[column].value_counts().plot(kind="bar")
            plt.title(f"Distribution of {column}")
            plt.xlabel("Categories")
            plt.ylabel("Count")
            plt.savefig("{column}_distribution.png")
            plt.show()

        # Summary Statistics for Labels
        print("\nSummary Statistics for Labels:")
        print(labels_df.describe())

        # Label Distribution
        print("\nLabel Distribution:")
        label_sums = labels_df.sum()
        print(label_sums)

        # Bar Plot for Label Distribution
        print("\nGenerating bar plot for label distribution...")
        label_sums.plot(kind="bar")
        plt.title("Label Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Count")
        plt.savefig("label_distribution.png")
        plt.show()

        # Checking Class Balance for Labels
        print("\nChecking class balance for labels...")
        total_samples = labels_df.shape[0]
        class_ratios = label_sums / total_samples
        for label, ratio in class_ratios.items():
            balance_status = "Balanced" if 0.4 <= ratio <= 0.6 else "Imbalanced"
            print(f"Label: {label}, Ratio: {ratio:.2f}, Status: {balance_status}")

        print("\nSummary statistics and visualizations for features, categorical features, and labels generated successfully.")

    def analyze_and_balance_labels(labels_df):
        """
        Analyzes label distribution and provides strategies for balancing multi-label datasets.

        Args:
            - labels_df (pd.DataFrame): DataFrame containing the binary multilabel target matrix.

        Returns:
            - None: Prints label and label-combination statistics.
        """

        # Analyze individual label distribution
        print("\nLabel Distribution:")
        label_sums = labels_df.sum()
        print(label_sums)

        # Identify unique label combinations
        label_combinations = [tuple(row) for row in labels_df.values]
        combination_counts = Counter(label_combinations)
        print("\nLabel Combination Distribution:")
        for combination, count in combination_counts.items():
            print(f"Combination {combination}: {count}")

        # Check for imbalance
        print("\nChecking Label Imbalance:")
        total_samples = labels_df.shape[0]
        for label, count in label_sums.items():
            balance_status = "Balanced" if 0.4 <= count / total_samples <= 0.6 else "Imbalanced"
            print(f"Label: {label}, Count: {count}, Status: {balance_status}")

        print("\nRecommendations for Balancing:")
        # Example recommendation based on thresholds
        threshold = total_samples * 0.1
        for combination, count in combination_counts.items():
            if count < threshold:
                print(f"Consider oversampling combination {combination}.")

    def visualize_data(protein_data):
        # Extract only the binary label columns.
        binary_labels = protein_data.iloc[:, 1:24]
        
        # Count occurrences of each label.
        label_counts = binary_labels.sum().sort_values(ascending=False)
        
        # Plot label frequency.
        plt.figure(figsize=(12, 6))
        label_counts.plot(kind='bar', color='skyblue')
        plt.title('Frequency of Binary Labels', fontsize=14)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Compute co-occurrence matrix.
        co_occurrence = binary_labels.T.dot(binary_labels)
        
        # Normalize for better visualization.
        co_occurrence_normalized = co_occurrence.div(co_occurrence.sum(axis=1), axis=0)
        
        # Plot heatmap of label co-occurrence.
        plt.figure(figsize=(12, 10))
        sns.heatmap(co_occurrence_normalized, annot=False, cmap="Blues", square=True, cbar=True)
        plt.title('Co-occurrence Heatmap of Binary Labels', fontsize=14)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Labels', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Count the number of labels assigned to each sample.
        multi_label_counts = binary_labels.sum(axis=1)
        
        # Plot histogram of multi-label counts.
        plt.figure(figsize=(10, 6))
        plt.hist(multi_label_counts, bins=range(1, multi_label_counts.max() + 2), color='skyblue', edgecolor='black', align='left')
        plt.title('Distribution of Multi-label Counts per Sample', fontsize=14)
        plt.xlabel('Number of Labels per Sample', fontsize=12)
        plt.ylabel('Frequency of Samples', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def remove_multi_label_samples(binary_multi_label_matrix, feature_matrix=None):
        """
        Removes all multi-label samples from a binary multi-label matrix.

        Args:
            - binary_multi_label_matrix (np.array or pd.DataFrame): 
            A binary multi-label matrix where each row corresponds to a sample 
            and each column corresponds to a label (1 for presence, 0 for absence).
            
            - feature_matrix (np.array or pd.DataFrame, optional):
            The feature matrix corresponding to the binary multi-label matrix. 
            If provided, it will also remove rows in the feature matrix corresponding 
            to multi-label samples.

        Returns:
            - filtered_labels (np.array): Binary matrix with only single-label samples.
            - filtered_features (np.array or None): Corresponding filtered feature matrix, if feature_matrix is provided. Otherwise, None.
        """
        # Calculate the sum of each row to count labels per sample.
        row_sums = np.sum(binary_multi_label_matrix, axis=1)

        # Identify single-label samples (rows with exactly one '1').
        single_label_indices = np.where(row_sums == 1)[0]

        # Filter the label matrix to include only single-label samples.
        filtered_labels = binary_multi_label_matrix[single_label_indices]

        # If feature_matrix is provided, filter it as well.
        if feature_matrix is not None:
            filtered_features = feature_matrix[single_label_indices]
            return filtered_labels, filtered_features

        return filtered_labels, None

    imported_data = pd.read_csv('pdb_data_no_dups.csv')
    # classification(imported_data,'rf',False)
    # classification(imported_data,'dt',False)
    classification(imported_data,'nn',False)