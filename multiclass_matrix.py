#!/usr/bin/env python3
"""multiclass_matrix.py: """

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import pandas as pd
import sys
import re

def create_multiclass_matrix(imported_data='pdb_data_no_dups.csv', matrix_creation_log='True'):
    # Select columns of interest.
    dataset = imported_data[['structureId','classification', 'macromoleculeType', 'residueCount', 'resolution',
                                'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue']]
    
    # Remove rows with missing values.
    dataset = dataset.dropna()
    
    # Keywords selected based on number of occurrence. To be considered a keyword, the word(s) must appear more than 1,000 times in the data set.
    keywords = ['structural','lyase','genomics','signal','transport','metal','membrane','isomerase','oxidoreductase','ligase','protein binding',
                'protein-binding','adhesion','chaperone','RNA','DNA','binding','rna binding','rna-binding','viru',
                'transferase','hydrolase','inhibitor','transcription','immune','genomics','regulator','regulation','viral','dna binding',
                'dna-binding']
    
    subclasses = ['hydrolase','transferase','oxidoreductase','dna_rna_binding','protein_binding','other_binding','inhibitor','transport',
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
        'rna binding': 'dna_rna_binding',
        'rna-binding': 'dna_rna_binding',
        'dna binding': 'dna_rna_binding',
        'dna-binding': 'dna_rna_binding',
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
    target_matrix = pd.DataFrame(0, index=dataset.index, columns=list(subclasses))
    
    # Safeguard structureId column.
    target_matrix.insert(0, 'structureId', dataset['structureId'])
    
    # Copy structureID into the target matrix.
    target_matrix['structureId'] = dataset['structureId']
    
    # Ensure the indices of dataset and target_matrix are aligned before populating the matrix.
    dataset.reset_index(drop=True, inplace=True)
    target_matrix.reset_index(drop=True, inplace=True)
    
    # Conditional logging for matrix creation.
    if matrix_creation_log == 'True':
        
        # Specify output file for matrix creation log.
        matrix_creation_file = 'matrix_creation_log.txt'
        
        # Open the file and redirect standard output to it.
        with open(matrix_creation_file, 'w') as f:
            sys.stdout = f
            
            # Populate the target matrix.
            for i, text in enumerate(dataset['classification']):
                structure_id = dataset.iloc[i]['structureId']  # Access structureId for the current row
                text_lower = text.lower()  # Convert to lowercase for case-insensitive matching
                
                # matrix_creation: Start processing a new row
                print(f"\nProcessing row {i}, StructureID: {structure_id}")
                print(f"Original text: {text_lower}")
                
                # Special case: Initialize flags for 'dna_rna_binding' and 'protein_binding'.
                triggers_dna_rna_binding = False
                triggers_protein_binding = False
                
                # Check for each keyword in the text
                for keyword, subclass in keyword_to_subclass.items():
                    if re.search(fr'{keyword}', text_lower): # Match keyword in text.
                        print(f"Match found: '{keyword}' -> Subclass: '{subclass}'")
                        if subclass == 'dna_rna_binding':
                            triggers_dna_rna_binding = True
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                            print(f"    Marking 'dna_rna_binding' (triggers_dna_rna_binding=True)")
                        elif subclass == 'protein_binding':
                            triggers_protein_binding = True
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                            print(f"    Marking 'protein_binding' (triggers_protein_binding=True)")
                        elif subclass != 'other_binding' and subclass != 'DNA': # Handles general subclasses.
                            target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                            print(f"    Marking subclass: '{subclass}'")
                
                # Special case: Handle 'other_binding' class ('binding' keyword trigger that is not mapped to dna_rna_binding or protein_binding).
                if re.search(r'binding',text_lower) and not triggers_dna_rna_binding and not triggers_protein_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('other_binding')] = 1
                    print(f"  Special case: Marking 'other_binding'")
                
                # Special case: Handle 'dna' keyword for 'DNA' class only if not already 'dna_rna_binding' class.
                if re.search(r'dna',text_lower) and not triggers_dna_rna_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('DNA')] = 1
                    print(f"  Special case: Marking 'DNA' (not triggered by 'dna_rna_binding')")
                
                # Special case: Handle 'rna' keyword for 'RNA' class only if not already 'dna_rna_binding' class.
                if re.search(r'rna',text_lower) and not triggers_dna_rna_binding:
                    target_matrix.iloc[i, target_matrix.columns.get_loc('RNA')] = 1
                    print(f"  Special case: Marking 'RNA' (not triggered by 'dna_rna_binding')")
                    
        # Reset standard output back to the console.
        sys.stdout = sys.__stdout__
        
        # Inform user of the matrix_creation log file.
        print(f"matrix_creation logs have been written to '{matrix_creation_file}'.")
    
    else:
        # Populate the target matrix.
        for i, text in enumerate(dataset['classification']):
            structure_id = dataset.iloc[i]['structureId']  # Access structureId for the current row.
            text_lower = text.lower()  # Convert to lowercase for case-insensitive matching.
            
            # matrix_creation: Start processing a new row.
            print(f"\nProcessing row {i}, StructureID: {structure_id}")
            print(f"Original text: {text_lower}")
            
            # Special case: Initialize flags for 'dna_rna_binding' and 'protein_binding'.
            triggers_dna_rna_binding = False
            triggers_protein_binding = False
            
            # Check for each keyword in the text.
            for keyword, subclass in keyword_to_subclass.items():
                if re.search(fr'{keyword}', text_lower): # Match keyword in text.
                    print(f"Match found: '{keyword}' -> Subclass: '{subclass}'")
                    
                    # Check for special case subclasses: dna_rna_binding, protein_binding, other_binding, DNA, RNA
                    if subclass == 'dna_rna_binding':
                        triggers_dna_rna_binding = True
                        target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                        
                    elif subclass == 'protein_binding':
                        triggers_protein_binding = True
                        target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
                        
                    elif subclass != 'other_binding' and subclass != 'DNA': # Handles general subclasses.
                        target_matrix.iloc[i, target_matrix.columns.get_loc(subclass)] = 1
            
            # Special case: Handle 'other_binding' class ('binding' keyword trigger that is not mapped to dna_rna_binding or protein_binding).
            # Prevents 'other_binding' being triggered when 'dna_rna_binding' or 'protein_binding' is triggered.
            if re.search(r'binding',text_lower) and not triggers_dna_rna_binding and not triggers_protein_binding:
                target_matrix.iloc[i, target_matrix.columns.get_loc('other_binding')] = 1
            
            # Special case: Handle 'dna' keyword for 'DNA' class only if not already 'dna_rna_binding' class.
            # Prevents 'dna' being triggered when 'dna_rna_binding' is triggered.
            if re.search(r'dna',text_lower) and not triggers_dna_rna_binding:
                target_matrix.iloc[i, target_matrix.columns.get_loc('DNA')] = 1
            
            # Special case: Handle 'rna' keyword for 'RNA' class only if not already 'dna_rna_binding' class.
            # Prevents 'rna' being triggered when 'dna_rna_binding' is triggered.
            if re.search(r'rna',text_lower) and not triggers_dna_rna_binding:
                target_matrix.iloc[i, target_matrix.columns.get_loc('RNA')] = 1
    
    # Remove rows that are all zeros (excluding 'structureId').
    non_zero_matrix = target_matrix.loc[(target_matrix.iloc[:, 1:] != 0).any(axis=1)]
    
    return (non_zero_matrix)