# 🍿 Netflix ML Hub: Recommendation & Classification Engine

A full-stack machine learning web application built to analyze, predict, and cluster Netflix content. This project demonstrates an end-to-end data science pipeline, from feature engineering and hyperparameter tuning to interactive web deployment. 

Developed as part of the **Auspify Data Science Internship**.

## Machine Learning Tasks & Technical Implementation

### Task 1: Content-Based Recommendation System
*   **Objective:** Suggest 5 similar movies or TV shows based on a user's selection by analyzing shared attributes.
*   **Approach:** Combined qualitative text features (genres, directors, countries, and ratings) into a single profile for each Netflix title.
*   **Technologies Used:** 
    *   `TfidfVectorizer` (Natural Language Processing) to convert the text attributes into mathematical matrices.
    *   `cosine_similarity` to measure the physical distance between those matrices, grouping titles that mathematically sit closest to each other.

### Task 2: Content Type Predictor (Binary Classification)
*   **Objective:** Predict whether a title is a "Movie" or a "TV Show" using its attributes.
*   **Approach:** Engineered custom numerical features from raw text (extracting minutes or season counts) and created a binary indicator for the presence of a director to teach the model how to classify the content.
*   **Technologies Used:** 
    *   `LabelEncoder` to translate text categories into numbers the models could read.
    *   `RandomForestClassifier` and `LogisticRegression` to generate the predictions.
    *   `train_test_split` to hide 20% of the dataset during training for accurate performance evaluation (achieving 99%+ accuracy).

### Task 3: Audience Rating Classification (Multi-Class Classification)
*   **Objective:** Predict the target audience (Kids, Teens, Adults) based on a title's genres, type, and duration.
*   **Approach:** Grouped 14 hyper-specific Netflix ratings into 3 broad target audiences to stabilize the model's learning environment. 
*   **Technologies Used:** 
    *   `CountVectorizer` to turn the top 150 text genres into binary columns (1 if the movie has the genre, 0 if it does not).
    *   `DecisionTreeClassifier` and `RandomForestClassifier` as the predictive algorithms.
    *   `GridSearchCV` for hyperparameter tuning, automatically testing multiple combinations of model settings to find the most accurate mathematical version.

### Task 4: Content Segmentation (Unsupervised Learning)
*   **Objective:** Autonomously group the Netflix library into 4 meaningful clusters without an answer key or labeled target variable.
*   **Approach:** Prepared a purely numeric dataset of release years, durations, and top genres, allowing the algorithm to find hidden patterns based on historical data.
*   **Technologies Used:** 
    *   `StandardScaler` to normalize the data so large numbers (like the year 2020) didn't overpower small numbers (like binary genre flags).
    *   `KMeans` to calculate distances and mathematically assign the 4 distinct clusters.
    *   `PCA` (Principal Component Analysis) to compress 13 different features down to 2 coordinates so they could be visualized on a screen.
    *   `plotly.express` to render the interactive, color-coded 2D scatter plot.

## 💻 Technology Stack

*   **Frontend:** Streamlit, HTML/CSS (via Markdown)
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Classification, Clustering, NLP, PCA, StandardScaler)
*   **Visualization:** Plotly Express

## ⚙️ How to Run Locally

1. Clone this repository to your local machine.
2. Ensure you have the dataset (`Dataset.csv`) in the root directory.
3. Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
