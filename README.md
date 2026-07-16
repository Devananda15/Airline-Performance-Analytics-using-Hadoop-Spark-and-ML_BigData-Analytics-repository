Airline Performance Analytics using Hadoop, Spark and Machine Learning
Project Overview
Airline Performance Analytics is a Big Data Analytics project that analyzes large-scale airline operational data using Apache Hadoop HDFS, Apache Spark, PySpark, and Streamlit. The system performs airline delay analysis, airport traffic analytics, route analysis, delay trend visualization, and machine learning-based flight clustering through an interactive dashboard.

Features
Airline Analytics
Average delay by airline
Flight cancellation analysis
Airline performance comparison
Airport Analytics
Top busiest airports
Airport traffic analysis
Delay distribution visualization
Route Analytics
Most popular flight routes
Interactive airport route map
Route traffic analysis
Delay Trend Analysis
Monthly delay trends
Day-wise delay patterns
Seasonal flight delay insights
Machine Learning Analytics
KMeans clustering using Spark MLlib
Flight behavior segmentation
Cluster distribution visualization
Technology Stack
Big Data Technologies
Apache Hadoop HDFS
Apache Spark
PySpark
Machine Learning
Spark MLlib
KMeans Clustering
Frontend & Visualization
Streamlit
Plotly
PyDeck
Programming Language
Python
Project Architecture
Dataset (Kaggle Airline Dataset) ↓ Apache Hadoop HDFS ↓ Apache Spark / PySpark ↓ Data Cleaning & Analytics ↓ Machine Learning (KMeans) ↓ Interactive Streamlit Dashboard

Dataset
Dataset Source: https://www.kaggle.com/datasets/usdot/flight-delays

Files Used:

flights.csv
airlines.csv
airports.csv
Place the downloaded files inside: dataset/

Installation
Clone Repository
git clone https://github.com/pavirb475/Airline-Performance-Analytics-using-Hadoop-Spark-and-ML_BigData-Analytics.git
cd Airline-Performance-Analytics-using-Hadoop-Spark-and-ML_BigData-Analytics
Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Hadoop HDFS Setup
Start HDFS:

start-dfs.sh
Create HDFS directory:

hdfs dfs -mkdir /airline_data
Upload datasets:

hdfs dfs -put dataset/flights.csv /airline_data/
hdfs dfs -put dataset/airlines.csv /airline_data/
hdfs dfs -put dataset/airports.csv /airline_data/
Verify upload:

hdfs dfs -ls /airline_data
Run the Dashboard
Activate environment:

source venv/bin/activate
Launch Streamlit:

streamlit run app.py
Open: http://localhost:8501

Machine Learning Workflow
Select flight features:
Arrival Delay
Departure Delay
Distance
Air Time
Create feature vectors using VectorAssembler.
Train KMeans clustering model.
Generate cluster predictions.
Visualize flight clusters and operational patterns.
