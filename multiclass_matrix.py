#!/usr/bin/env python3
"""multiclass_matrix.py: Contains module that creates a multi-class binary matrix based on the classification data from a protein data set."""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import pandas as pd
import sys
import re

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
    data_set = imported_data[['structureId','classification', 'macromoleculeType', 'residueCount', 'resolution',
                                'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue']]
    
    # Remove rows with missing values.
    data_set = data_set.dropna()
    
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
    target_matrix = pd.DataFrame(0, index=data_set.index, columns=list(subclasses))
    
    # Safeguard structureId column.
    target_matrix.insert(0, 'structureId', data_set['structureId'])
    
    # Copy structureID into the target matrix.
    target_matrix['structureId'] = data_set['structureId']
    
    # Ensure the indices of data_set and target_matrix are aligned before populating the matrix.
    data_set.reset_index(drop=True, inplace=True)
    target_matrix.reset_index(drop=True, inplace=True)
    
    # Conditional logging for matrix creation.
    if matrix_creation_log == 'True':
        
        # Specify output file for matrix creation log.
        matrix_creation_file = 'matrix_creation_log.txt'
        
        # Open the file and redirect standard output to it.
        with open(matrix_creation_file, 'w') as f:
            sys.stdout = f
            
            # Populate the target matrix.
            for i, text in enumerate(data_set['classification']):
                structure_id = data_set.iloc[i]['structureId']  # Access structureId for the current row
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
        for i, text in enumerate(data_set['classification']):
            structure_id = data_set.iloc[i]['structureId']  # Access structureId for the current row.
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
    if 'structureId' in non_zero_matrix.columns and 'structureId' in data_set.columns:
        matrix_data_set = non_zero_matrix.merge(
            data_set[['structureId'] + columns_to_add],  # Select columns to merge.
            on='structureId',  # Merge on the structureId column.
            how='left'  # Keep all rows in non_zero_matrix.
        )
    else:
        print("Error: 'structureId' must exist in both non_zero_matrix and data set.")
    
    # Display the updated matrix_data_set structure.
    print(matrix_data_set.head())
    print(matrix_data_set.columns)
    matrix_data_set.to_csv('matrix_data_set.csv', index=True)
    
    return (matrix_data_set)

# Test run.
imported_data = pd.read_csv('pdb_data_no_dups.csv')
create_multiclass_matrix(imported_data)

# --- Ended up commenting out this method since I realized that the macromolecule_type column should work without being modified.

# def normalize_macromolecule_type(imported_data='pdb_data_no_dups.csv'):
#     # Import data into Pandas data set.
#     imported_data = pd.read_csv(imported_data)
    
#     # Select columns of interest.
#     data_set = imported_data[['structureId','classification', 'macromoleculeType', 'residueCount', 'resolution',
#                                 'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue']]
    
#     # Remove rows with missing values.
#     data_set = data_set.dropna()
    
#     # Print all unique values in the 'macromoleculeType' column
#     unique_values = data_set['macromoleculeType'].unique()
#     print("Unique values in 'macromoleculeType':", unique_values)