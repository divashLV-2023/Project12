import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder


# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Team 12 : Health CLustering", layout="wide")

# Header
st.header("Patient Health Risk Stratification")


# Load data
@st.cache_resource()
def load_data():
    dataa = pd.read_csv(
        'https://raw.githubusercontent.com/divashLV-2023/Project12/main/heart_attack_prediction_dataset%20(2).csv')
    return dataa.copy()


df = load_data()


# Initial data insights
st.write("### Initial Data Insights:")
st.write("Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:")
st.write(df.describe())  # Display summary statistics

# Sidebar options
st.sidebar.title("Options")
st.sidebar.write("Select options to view analysis:")
segmentation_type = st.sidebar.selectbox('Select disease type',
                                         ['Heart Attack','Diabetes', 'Obesity',  'Hyper Tension'])



if segmentation_type == 'Heart Attack':
    show_clusters = st.sidebar.checkbox("Show Cluster Visualization")
    show_avg_values = st.sidebar.checkbox("Show Average Risk Factor Values")
    show_silhouette_score = st.sidebar.checkbox("Show Silhouette Score")
    show_davies_bouldin = st.sidebar.checkbox("Show Davies-Bouldin Index")
    show_Calinski_Harabasz= st.sidebar.checkbox("Show Calinski_Harabasz Index")

    # Header
    st.header("Heart attack:")
    st.write("#### Visualizations and output:")
    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/divashLV-2023/Project12/main/heart_attack_prediction_dataset%20(2).csv')

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Diet'] = le.fit_transform(df['Diet'])
    df['Country'] = le.fit_transform(df['Country'])
    df['Continent'] = le.fit_transform(df['Continent'])
    df['Hemisphere'] = le.fit_transform(df['Hemisphere'])


    # Select relevant features for clustering
    features = ['Age', 'Sex', 'Systolic_Blood_Pressure ', 'Diastolic_Blood_Pressure',
                'Smoking', 'Diet', 'BMI', 'Alcohol_Consumption',
                'Exercise_Hours_Per_Week', 'Sedentary_Hours_Per_Day', 'Stress_Level', 'Previous_Heart_Problems',
                'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day']

    # Extract feature data
    X = df[features]

    # Preprocess: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Choosing the number of clusters
    n_clusters = 2

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=41)
    kmeans.fit(X_scaled)

    # Add cluster labels to DataFrame
    df['Cluster'] = kmeans.labels_

    # Display cluster visualization if selected
    if show_clusters:
        st.header("K-means Clustering Visualization (PCA)")

        # Plot clusters
        fig, ax = plt.subplots()
        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(X_pca[df['Cluster'] == cluster][:, 0], X_pca[df['Cluster'] == cluster][:, 1],label=f'Cluster {cluster}')

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('K-means Clustering of Patients (PCA)')
        ax.legend()

        # Display the plot using Streamlit
        st.pyplot(fig)


    # Calculate and display average risk factor values if selected
    if show_avg_values:
        st.header("Average Risk Factor Values for Each Cluster")
        cluster_avg = df.groupby('Cluster')[['Age', 'Cholesterol']].mean()

        # Print average values for each cluster
        st.write("##### Average values of risk factors for each cluster:")
        st.write(round(cluster_avg,2))

        # Compare the average values between clusters
        if cluster_avg.loc[0, 'Cholesterol'] > cluster_avg.loc[1, 'Cholesterol']:
            st.write("Cluster 0 has higher average cholesterol levels.")
        else:
            st.write("Cluster 1 has higher average cholesterol levels.")

        if cluster_avg.loc[0, 'Age'] > cluster_avg.loc[1, 'Age']:
            st.write("Cluster 0 has higher average age.")
        else:
            st.write("Cluster 1 has higher average age.")

    # Calculate and display silhouette score if selected
    if show_silhouette_score:
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        st.write("##### Silhouette Score:", round(silhouette_avg,2))

    # Calculate and display Davies-Bouldin index if selected
    if show_davies_bouldin:
        davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
        st.write("##### Davies-Bouldin Index:", round(davies_bouldin,2))

    # Calinski-Harabasz Index
    if show_Calinski_Harabasz:
        calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
        st.write("##### Calinski-Harabasz Index:", round(calinski_harabasz,2))



elif segmentation_type == 'Diabetes':
    show_clusters = st.sidebar.checkbox("Show Cluster Visualization")
    show_avg_values = st.sidebar.checkbox("Show Average Risk Factor Values")
    show_silhouette_score = st.sidebar.checkbox("Show Silhouette Score")
    show_davies_bouldin = st.sidebar.checkbox("Show Davies-Bouldin Index")
    show_Calinski_Harabasz= st.sidebar.checkbox("Show Calinski_Harabasz Index")

    # Header
    st.header("Diabetes:")
    st.write("#### Visualizations and output:")

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/divashLV-2023/Project12/main/heart_attack_prediction_dataset%20(2).csv')
   # df['Blood_Pressure'] = df['Blood_Pressure'].str.replace('/', '.')
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Diet'] = le.fit_transform(df['Diet'])
    df['Country'] = le.fit_transform(df['Country'])
    df['Continent'] = le.fit_transform(df['Continent'])
    df['Hemisphere'] = le.fit_transform(df['Hemisphere'])

    # Selecting relevant features for clustering
    features = ['Diet', 'Diabetes', 'Sedentary_Hours_Per_Day', 'Stress_Level', 'Active_hours_per_day',
                'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day']

    # Extracting feature data
    X = df[features]

    # Preprocessing: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Choosing the number of clusters (you can experiment with different values)
    n_clusters = 2

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Adding cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Display cluster visualization if selected
    if show_clusters:
        st.header("K-means Clustering Visualization (PCA)")

        # Plot clusters
        fig, ax = plt.subplots()
        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(X_pca[df['Cluster'] == cluster][:, 0], X_pca[df['Cluster'] == cluster][:, 1],
                       label=f'Cluster {cluster}')

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('K-means Clustering of Patients (PCA)')
        ax.legend()
        st.pyplot(fig)


    # Calculate and display average risk factor values if selected
    if show_avg_values:
        st.header("Average Risk Factor Values for Each Cluster")
        cluster_avg = df.groupby('Cluster')[['Age', 'Cholesterol']].mean()

        # Print average values for each cluster
        st.write("##### Average values of risk factors for each cluster:")
        st.write(round(cluster_avg,2))

        # Compare average values between clusters
        if cluster_avg.loc[0, 'Cholesterol'] > cluster_avg.loc[1, 'Cholesterol']:
            st.write("Cluster 0 has higher average cholesterol levels.")
        else:
            st.write("Cluster 1 has higher average cholesterol levels.")

        if cluster_avg.loc[0, 'Age'] > cluster_avg.loc[1, 'Age']:
            st.write("Cluster 0 has higher average age.")
        else:
            st.write("Cluster 1 has higher average age.")

    # Calculate and display silhouette score if selected
    if show_silhouette_score:
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        st.write("##### Silhouette Score:", round(silhouette_avg,2))

    # Calculate and display Davies-Bouldin index if selected
    if show_davies_bouldin:
        davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
        st.write("##### Davies-Bouldin Index:", round(davies_bouldin,2))

    # Calinski-Harabasz Index
    if show_Calinski_Harabasz:
        calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
        st.write("##### Calinski-Harabasz Index:", round(calinski_harabasz,2))



elif segmentation_type == 'Obesity':
    show_clusters = st.sidebar.checkbox("Show Cluster Visualization")
    show_avg_values = st.sidebar.checkbox("Show Average Risk Factor Values")
    show_silhouette_score = st.sidebar.checkbox("Show Silhouette Score")
    show_davies_bouldin = st.sidebar.checkbox("Show Davies-Bouldin Index")
    show_Calinski_Harabasz= st.sidebar.checkbox("Show Calinski_Harabasz Index")

    # Header
    st.header("Obesity:")
    st.write("#### Visualizations and output:")

    # Load data
    df1 = pd.read_csv('https://raw.githubusercontent.com/divashLV-2023/Project12/main/heart_attack_prediction_dataset%20(2).csv')
    le = LabelEncoder()
    df1['Sex'] = le.fit_transform(df1['Sex'])
    df1['Diet'] = le.fit_transform(df1['Diet'])
    df1['Country'] = le.fit_transform(df1['Country'])
    df1['Continent'] = le.fit_transform(df1['Continent'])
    df1['Hemisphere'] = le.fit_transform(df1['Hemisphere'])

    # Selecting relevant features for clustering
    features = ['Age', 'Sex', 'Diabetes', 'Diet', 'BMI', 'Alcohol_Consumption',
                'Exercise_Hours_Per_Week', 'Sedentary_Hours_Per_Day', 'Stress_Level', 'Smoking',
                'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day', 'Active_hours_per_day']

    # Extracting feature data
    X = df1[features]

    # Preprocessing: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Choosing the number of clusters (you can experiment with different values)
    n_clusters = 2

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=40)
    kmeans.fit(X_scaled)

    df1['Cluster'] = kmeans.labels_

    # Display cluster visualization if selected
    if show_clusters:
        st.header("K-means Clustering Visualization (PCA)")

        fig, ax = plt.subplots()
        for cluster in range(n_clusters):
            cluster_data = df1[df1['Cluster'] == cluster]
            ax.scatter(X_pca[df1['Cluster'] == cluster][:, 0], X_pca[df1['Cluster'] == cluster][:, 1],
                       label=f'Cluster {cluster}')

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('K-means Clustering of Patients (PCA)')
        ax.legend()
        st.pyplot(fig)


    # Calculate and display average risk factor values if selected
    if show_avg_values:
        st.header("Average Risk Factor Values for Each Cluster")
        cluster_avg = df1.groupby('Cluster')[['Age', 'Cholesterol']].mean()

        # Print average values for each cluster
        st.write("##### Average values of risk factors for each cluster:")
        st.write(round(cluster_avg,2))

        # Compare average values between clusters
        if cluster_avg.loc[0, 'Cholesterol'] > cluster_avg.loc[1, 'Cholesterol']:
            st.write("Cluster 0 has higher average cholesterol levels.")
        else:
            st.write("Cluster 1 has higher average cholesterol levels.")

        if cluster_avg.loc[0, 'Age'] > cluster_avg.loc[1, 'Age']:
            st.write("Cluster 0 has higher average age.")
        else:
            st.write("Cluster 1 has higher average age.")

    # Calculate and display silhouette score if selected
    if show_silhouette_score:
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        st.write("##### Silhouette Score:", round(silhouette_avg,2))

    # Calculate and display Davies-Bouldin index if selected
    if show_davies_bouldin:
        davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
        st.write("##### Davies-Bouldin Index:", round(davies_bouldin,2))

    # Calinski-Harabasz Index
    if show_Calinski_Harabasz:
        calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
        st.write("##### Calinski-Harabasz Index:", round(calinski_harabasz,2))



elif segmentation_type == 'Hyper Tension':
    show_clusters = st.sidebar.checkbox("Show Cluster Visualization")
    show_avg_values = st.sidebar.checkbox("Show Average Risk Factor Values")
    show_silhouette_score = st.sidebar.checkbox("Show Silhouette Score")
    show_davies_bouldin = st.sidebar.checkbox("Show Davies-Bouldin Index")
    show_Calinski_Harabasz= st.sidebar.checkbox("Show Calinski_Harabasz Index")

    # Header
    st.header("Hyper Tension:")
    st.write("#### Visualizations and output:")

    # Load data
    df4 = pd.read_csv('https://raw.githubusercontent.com/divashLV-2023/Project12/main/heart_attack_prediction_dataset%20(2).csv')
    #df4['Blood_Pressure'] = df4['Blood_Pressure'].str.replace('/', '.')
    le = LabelEncoder()
    df4['Sex'] = le.fit_transform(df4['Sex'])
    df4['Diet'] = le.fit_transform(df4['Diet'])
    df4['Country'] = le.fit_transform(df4['Country'])
    df4['Continent'] = le.fit_transform(df4['Continent'])
    df4['Hemisphere'] = le.fit_transform(df4['Hemisphere'])

    # Selecting relevant features for clustering
    features = ['Age', 'Sex', 'Systolic_Blood_Pressure ', 'Diastolic_Blood_Pressure', 'Stress_Level',
                'Physical_Activity_Days_Per_Week', 'BMI', 'Diet',
                'Sleep_Hours_Per_Day', 'Smoking', 'Alcohol_Consumption']

    # Extracting feature data
    X = df4[features]

    # Preprocessing: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Choosing the number of clusters (you can experiment with different values)
    n_clusters = 2

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=45)
    kmeans.fit(X_scaled)

    # Adding cluster labels to the DataFrame
    df4['Cluster'] = kmeans.labels_


    # Display cluster visualization if selected
    if show_clusters:
        st.header("K-means Clustering Visualization (PCA)")

        # Plot clusters
        fig, ax = plt.subplots()
        for cluster in range(n_clusters):
            cluster_data = df4[df4['Cluster'] == cluster]
            ax.scatter(X_pca[df4['Cluster'] == cluster][:, 0], X_pca[df4['Cluster'] == cluster][:, 1],
                       label=f'Cluster {cluster}')

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('K-means Clustering of Patients (PCA)')
        ax.legend()
        st.pyplot(fig)  # Display the plot in Streamlit


    # Calculate and display average risk factor values if selected
    if show_avg_values:
        cluster_avg = df4.groupby('Cluster')[['Age', 'Cholesterol']].mean()

        # Print the average values for each cluster
        st.write("##### Average values of risk factors for each cluster:")
        st.write(round(cluster_avg,2))

        # Compare the average values between clusters
        if cluster_avg.loc[0, 'Cholesterol'] > cluster_avg.loc[1, 'Cholesterol']:
            st.write("Cluster 0 has higher average cholesterol levels.")
        else:
            st.write("Cluster 1 has higher average cholesterol levels.")

        if cluster_avg.loc[0, 'Age'] > cluster_avg.loc[1, 'Age']:
            st.write("Cluster 0 has higher average age.")
        else:
            st.write("Cluster 1 has higher average age.")

    # Calculate and display silhouette score if selected
    if show_silhouette_score:
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        st.write("##### Silhouette Score:", round(silhouette_avg,2))

    # Calculate and display Davies-Bouldin index if selected
    if show_davies_bouldin:
        davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
        st.write("##### Davies-Bouldin Index:", round(davies_bouldin,2))

    # Calinski-Harabasz Index
    if show_Calinski_Harabasz:
        calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
        st.write("##### Calinski-Harabasz Index:", round(calinski_harabasz,2))


st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Team 12 ")
st.sidebar.markdown("- Sravanthi")
st.sidebar.markdown("- Geethanjali")
st.sidebar.markdown("- Garvit")
st.sidebar.markdown("- Ashish")
