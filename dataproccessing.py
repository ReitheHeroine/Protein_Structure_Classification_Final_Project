#!/usr/bin/env python3
"""dataprocessing.py: *** """

__author__ = "Reina Hastings, Eugenio Casta"
__email__ = "reinahastings13@gmail.com"

import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import re
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

def row_validation(row):
    for cell in row:
        # Check for NaN, None, or blank strings and filter them out
        if cell is None or cell == '' or (isinstance(cell, float) and np.isnan(cell)):
            return False
    return True

def classification(imported_data, model):
    # x0 = imported_data['macromoleculeType']
    # x1 = imported_data['residueCount']
    # x2 = imported_data['structureMolecularWeight']
    # x3 = imported_data['densityMatthews']
    # x4 = imported_data['densityPercentSol']
    # x5 = imported_data['phValue']
    
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
    
    # # Convert sample_class from pd to np array.
    # sample_class = sample_class.to_numpy()
    
    # dataset = np.array([x0,x1,x2,x3,x4,x5,sample_class], dtype=object)
    # dataset = dataset.T
    
    # # Remove rows that fail row_validation check (contains NaN, None, or '').
    # dataset = np.array([row for row in dataset if row_validation(row)], dtype=object)
    
    # Keep copy of structureId column.
    structure_IDs = protein_data['structureId']
    
    # Drop the 'structureId' column from data set before training.
    protein_data = protein_data.drop(columns=['structureId'])
    
    # encoder = LabelEncoder()
    # for i in range(dataset.shape[1]):
    #     if dataset[:, i].dtype == object:  # Check if column has strings
    #         dataset[:, i] = encoder.fit_transform(dataset[:, i])
    
    # One-hot encoding of 'macromoleculeType' variable since it is nominal.
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(protein_data[['macromoleculeType']])
    
    # Turn encoded from NumPy array to Pandas array.
    feature_names = encoder.get_feature_names_out(['macromoleculeType'])
    encoded = pd.DataFrame(encoded, columns=feature_names)
    
    # # Check if df is a DataFrame
    # is_dataframe = isinstance(encoded, pd.DataFrame)
    # print(is_dataframe)  # This will print: True
    
    # Optional: Save encoded data to a csv file to visually inspect.
    encoded.to_csv('encoded.csv', index=True)
    
    # Drop 'macromoleculeType' column from protein_data.
    protein_data = protein_data.drop('macromoleculeType', axis=1)
    
    # Add encoded data frame containing the one-hot encoded 'macromoleculeType' data to protein_data.
    # *** Note to self: If I have issues reassigning the structureIds back onto the df, check this line since the indexing could have been tampered with. ***
    # **************** Current issue identified, encoded becomes blank when added to protein_data. *************************
    protein_data = pd.concat([protein_data,encoded], ignore_index=False)
    
    # Optional: Save protein_data to a csv file to visually inspect.
    protein_data.to_csv('protein_data.csv', index=True)
    
    print('Protein data: ')
    print(protein_data.shape)
    
    # dataset = dataset.astype(np.float32)
    
    # label_matrix includes the first 24 columns which make up the classifier binary matrix.
    label_matrix = protein_data.iloc[:, 0:24]
    
    # features includes the rest of the data frame after the first 24 columns.
    features = protein_data.iloc[:, 24:]
    
    # Optional: Save features and label_matrix to a csv files to visually inspect.
    label_matrix.to_csv('label_matrix.csv', index=True)
    features.to_csv('features.csv', index=True)
    
    if model == 'rf':
        random_forest(features, label_matrix)
    elif model == 'dt':
        decision_tree(features, label_matrix)
    elif model == 'nn':
        neural_network(features,label_matrix)

def random_forest(features, labels):
    # First Split: Train (80%) and validation + test (20%) sets.
    X_train, X_test_val, y_train, y_test_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Second Split: Train (50%) and validation (50%) sets.
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    
    # Total data set split into train (80%), validation (10%), and test (10%).
    
    # Optional: Check shape of train, validation, and test sets.
    # print('Test/train/val: ')
    # print(X_test.shape)
    # print(y_test.shape)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    
    print(y_train)
    
    # Initialize base model.
    # n_estimaors = 100 -> 100 trees
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None)
    
    # Wrap model with MultiOutputClassifier.
    multi_target_rf = MultiOutputClassifier(rf_model, n_jobs=-1)
    
    # Train the model.
    multi_target_rf.fit(X_train,y_train)

    y_pred = multi_target_rf.predict(X_test)

    c_matrix(y_test, y_pred)

def decision_tree(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    c_matrix(y_test, y_pred)

def neural_network(data, labels):
    data = np.expand_dims(data, axis=-1)
    num_classes = len(np.unique(labels))

    #Establishes training/testing data
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.25)

    #Defines the cnn model to be used and the layers that will be used in the process
    model = Sequential([
        Flatten(input_shape=(data.shape[1],data.shape[2])),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    #Defines the optimizer, the learning rate, and compiles the model using given parameters
    opt = Adam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    #Displays live model data processing and fits the training data
    model.summary()
    model.fit(X_train, y_train, epochs=15)
    print("Finish training")

    #Displays test label and compares it to label the cnn predicted
    y_pred_prob = model.predict(X_test)
    print("y_test:", y_test)
    print("y_pred_prob:", y_pred_prob)

    y_pred, truth = [], []
    for prob in y_pred_prob:
        y_pred.append(np.argmax(prob))
    for prob in y_test:
        truth.append(np.argmax(prob))

    #Displays accuracy of the model given the performance across all samples
    c_matrix(truth, y_pred)

def c_matrix(y_true, y_pred):
    confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted',zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted',zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted',zero_division=1)
    print(f"Accuracy: {accuracy * 100}%")
    print(f"Precision: {precision * 100}%")
    print(f"Recall: {recall * 100}%")
    print(f"F1 Score: {f1 * 100}%")

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

imported_data = pd.read_csv('pdb_data_no_dups.csv')
classification(imported_data,'rf')