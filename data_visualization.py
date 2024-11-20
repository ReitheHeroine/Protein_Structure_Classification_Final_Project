#!/miniconda3/envs/class/bin/python

"""Homework #1 for CS-653"""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import os
from scipy.spatial import distance
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
import seaborn as sea
import numpy as np


def main():
    # Import usda_data.csv dataset.
    usda_data = pd.read_csv('usda_data.csv', sep=',', header=0)
    
    # Remove all columns except cultivarName, alphaAcidsAverage, betaAcidsAverage, and cohumuloneAverage.
    columns_of_interest = ['cultivarName', 'alphaAcidsAverage', 'betaAcidsAverage', 'cohumuloneAverage']
    usda_data = usda_data[columns_of_interest]
    
    print(len(usda_data))
    
    # Drop samples with missing data.
    usda_data = usda_data.dropna()
    
    print(len(usda_data))
    print(usda_data)
    
    # Create histogram of the data set.
    create_histo(usda_data, 'Histogram')
    
    return
    
    # Question 3: Generate a scatter matrix for the attributes: fixed acidity, volatile acidity, citric acid, and residual sugar
    
    # Create paired box plots for each attributes, one from each data set
    create_box_plots(red_wine_quality, 'red_wine', white_wine_quality, 'white_wine')
    
    # Create scatter matrices
    create_scatter(red_wine_quality, 'red_wine')
    create_scatter(white_wine_quality, 'white_wine')
    
    # Question 4: Visualize between-sample similarity matrices over each of the two datasets
    distance_heat_maps(red_wine_quality, 'red_wine')
    distance_heat_maps(white_wine_quality, 'white_wine')

# Creates a histogram of all attributes in a given data set
def create_histo(df, df_name):
    # Iterating through the attributes in the data frame
    for col in df.columns:
        # Creating a histogram for the attribute
        plt.hist(df[col], bins=10, edgecolor='black')
        plt.title(f'Histogram for {col}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        
        # Check if a histograms folder exists in local file, if not, create one
        if not os.path.exists('histograms'):
            os.mkdir('histograms')
        
        # Saving the histogram to histograms file
        plt.savefig(f'histograms/{df_name}_{col}_histo.png')
        
        # Clearing the histogram
        plt.clf()


# Creates side by side box plots of all attributes in two given data sets
def create_box_plots(df_A, df_name_A, df_B, df_name_B):
    # Assuming both datasets have the same attributes
    attributes = df_A.columns

    # Loop through attributes to create matching box plots for each one
    for attribute in attributes:
        # Create a figure with two subplots (side-by-side)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Boxplot for Dataset A
        ax[0].boxplot(df_A[attribute])
        ax[0].set_title(f'{attribute} from {df_name_A}')
        ax[0].set_ylabel('Values')

        # Boxplot for Dataset B
        ax[1].boxplot(df_B[attribute])
        ax[1].set_title(f'{attribute} from {df_name_B}')
        ax[1].set_ylabel('Values')

        # Set a common title for the whole figure
        plt.suptitle(f'Comparison of {attribute}')

        # Check if a box_plots folder exists in local file, if not, create one
        if not os.path.exists('box_plots'):
            os.mkdir('box_plots')
            
        # Saving the box plot to box_plots file
        plt.savefig(f'box_plots/{df_name_A}_vs_{df_name_B}_{attribute}_box.png')
        
        # Clearing the box plot
        plt.clf()


# Creates scatter matrices of the following attributes in a given data set: fixed acidity, volatile acidity, citric acid, and residual sugar
def create_scatter(df, df_name):
    # Selecting for the attributes of interest: fixed acidity, volatile acidity, citric acid, and residual sugar
    selected_attributes_df = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']]
    
    # Scatter plot for Dataset A
    scatter_matrix(selected_attributes_df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    
    # Set a common title for the whole figure
    plt.suptitle(f'Comparison of attributes in {df_name}')
    
    # Check if a scatter_matrices folder exists in local file, if not, create one
    if not os.path.exists('scatter_matrices'):
        os.mkdir('scatter_matrices')
        
    # Saving the scatter plot to scatter_matrices file
    plt.savefig(f'scatter_matrices/{df_name}_scatter_matrix.png')
    
    # Clearing the scatter plot
    plt.clf()


# Measures Euclidean, Manhattan, Minkowski, Mahalanobis, correlation, distances, a linear combination of Euclidean and Minkowski distances, and cosine similarity between all possible sample combinations in given data set
def distance_heat_maps(df, df_name):
    # Sort the data set by quality
    df_sorted = df.sort_values(by='quality')
    
    # Extract required attributes from the data set
    attributes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates']
    
    df_sorted = df_sorted[attributes]
    
    # # Create data set that only includes the first 50 samples for efficient testing
    # df_sorted = df_sorted.iloc[:50]
    
    # Euclidean distance
    euclidean_distances = distance.cdist(df_sorted, df_sorted, metric='euclidean')
    
    # Minkowski distance (r=1)
    minkowski_r1_distances = distance.cdist(df_sorted, df_sorted, metric='minkowski', p=1)

    # Minkowski distance (r=3)
    minkowski_r3_distances = distance.cdist(df_sorted, df_sorted, metric='minkowski', p=3)

    # Minkowski distance (r=5)
    minkowski_r5_distances = distance.cdist(df_sorted, df_sorted, metric='minkowski', p=5)

    # Cosine similarity
    cosine_similarities = cosine_similarity(df_sorted)
    
    # Mahalanobis distance (requires inverse covariance matrix to correct for correlation)
    inv_cov_matrix = np.linalg.inv(np.cov(df_sorted.T))
    mahalanobis_distances = distance.cdist(df_sorted, df_sorted, metric='mahalanobis', VI=inv_cov_matrix)
    
    # Correlation distance
    correlation_distances = distance.cdist(df_sorted, df_sorted, metric='correlation')

    # Linear combination of Euclidean and Minkowski (r=1)
    linear_combination_distances = 0.5 * euclidean_distances + 0.5 * minkowski_r1_distances
    
    # Plot heat maps using all distance metrics
    plot_heatmap(euclidean_distances, f'{df_name}_euclidean_distances')
    plot_heatmap(minkowski_r1_distances, f'{df_name}_minkowski_distances_r=1')
    plot_heatmap(minkowski_r3_distances, f'{df_name}_minkowski_distances_r=3')
    plot_heatmap(minkowski_r5_distances, f'{df_name}_minkowski_distances_r=5')
    plot_heatmap(cosine_similarities, f'{df_name}_cosine_similarities')
    plot_heatmap(mahalanobis_distances, f'{df_name}_mahalanobis_distances')
    plot_heatmap(correlation_distances, f'{df_name}_correlation_distances')
    plot_heatmap(linear_combination_distances, f'{df_name}_linear_combination_distances')
    
    # Break down the data into smaller subsets to create more readable heat maps
    quality_groups = df.groupby('quality')
    
    # Iterate through each group of data clustered by quality and calculate distance metrics
    for quality, group in quality_groups:
        # Drop the 'quality' attribute from the group
        group_without_quality = group.drop(columns=['quality'])

        # Euclidean distance
        quality_euclidean_distances = distance.cdist(group_without_quality, group_without_quality, metric='euclidean')
        
        # Minkowski distance (r=1)
        quality_minkowski_r1_distances = distance.cdist(group_without_quality, group_without_quality, metric='minkowski', p=1)

        # Minkowski distance (r=3)
        quality_minkowski_r3_distances = distance.cdist(group_without_quality, group_without_quality, metric='minkowski', p=3)

        # Minkowski distance (r=5)
        quality_minkowski_r5_distances = distance.cdist(group_without_quality, group_without_quality, metric='minkowski', p=5)

        # Cosine similarity
        quality_cosine_similarities = cosine_similarity(group_without_quality)
        
        # Mahalanobis distance (requires inverse covariance matrix to correct for correlation)
        quality_inv_cov_matrix = np.linalg.inv(np.cov(group_without_quality.T))
        quality_mahalanobis_distances = distance.cdist(group_without_quality, group_without_quality, metric='mahalanobis', VI=inv_cov_matrix)
        
        # Correlation distance
        quality_correlation_distances = distance.cdist(group_without_quality, group_without_quality, metric='correlation')

        # Linear combination of Euclidean and Minkowski (r=1)
        quality_linear_combination_distances = 0.5 * quality_euclidean_distances + 0.5 * quality_minkowski_r1_distances    

        # Plot quality subset heat maps using all distance metrics
        plot_heatmap(quality_euclidean_distances, f'{df_name}_quality_{quality}_euclidean_distances')
        plot_heatmap(quality_minkowski_r1_distances, f'{df_name}__quality_{quality}_minkowski_distances_r=1')
        plot_heatmap(quality_minkowski_r3_distances, f'{df_name}__quality_{quality}_minkowski_distances_r=3')
        plot_heatmap(quality_minkowski_r5_distances, f'{df_name}__quality_{quality}_minkowski_distances_r=5')
        plot_heatmap(quality_cosine_similarities, f'{df_name}__quality_{quality}_cosine_similarities')
        plot_heatmap(quality_mahalanobis_distances, f'{df_name}__quality_{quality}_mahalanobis_distances')
        plot_heatmap(quality_correlation_distances, f'{df_name}__quality_{quality}_correlation_distances')
        plot_heatmap(quality_linear_combination_distances, f'{df_name}__quality_{quality}_linear_combination_distances')


# Create heat maps using given distance metric
def plot_heatmap(distance, title):
    # Plot distance heatmap
    plt.figure(figsize=(15, 15))
    sea.heatmap(distance, cmap="coolwarm")
    plt.title(title)
    plt.show()
    
    # Check if a heat_maps folder exists in local file, if not, create one
    if not os.path.exists('heat_maps'):
        os.mkdir('heat_maps')
        
    # Saving the heat map plot to heat_maps file
    plt.savefig(f'heat_maps/{title}_heat_map.png')
    
    # Clearing the heat map plot
    plt.clf()

if __name__ == "__main__":
        main()