import pandas as pd 
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer   #converts raw text into a matrix
from sklearn.metrics.pairwise import cosine_similarity   #calculates the sim between two vectors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression    #makes decisions using statistical models
from sklearn.preprocessing import LabelEncoder, StandardScaler   #translates texts to numerical IDs
from sklearn.ensemble import RandomForestClassifier   #makes decisions using multiple decision trees
from sklearn.metrics import accuracy_score   
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA    #squishes high dim data down to 2D for graphs
import plotly.express as px     #for interactive web graphs

st.set_page_config(page_title='Netflix ML project', layout='wide')
st.sidebar.title('Tasks')
page = st.sidebar.radio('Select a task:', [
    'Task 1: Recommendation system', 
    'Task 2: Content prediction', 
    'Task 3: Rating Classification',
    'Task 4: Content Segmentation'
])

#---------------------------------- TASK 1 ----------------------------------
#Recommendation system 

if page=='Task 1: Recommendation system':
    st.title('Movie Recommender System')
    st.write('Find similar movies and TV shows based on genres, directors and ratings.')

#1.Preparing the data

    @st.cache_data     #makes the function run once so its faster when called again
    def load_data():
       df=pd.read_csv("Dataset.csv")
       features=['type','director','country','rating','listed_in']   #only relevant features we need

       for feature in features:  
          df[feature]=df[feature].fillna('')    #loop through each feature and remove missing values

       def combine_features(row):
           return f"{row['type']} {row['director']} {row['country']} {row['rating']} {row['listed_in']}"    
#takes the row and combines all the features in one string

       df['combined_features']=df.apply(combine_features, axis=1)   #applying the func to each row 

#2.Converting text

       tfidf= TfidfVectorizer(stop_words='english')    #remove english stop words
       tfidf_matrix=tfidf.fit_transform(df['combined_features'])    #learns the vocabulary and translates to matrix

#3.Calculate content similarity

       cosine_sim=cosine_similarity(tfidf_matrix, tfidf_matrix)    #calc sim between all the rows in the matrix
       return df,cosine_sim
 
    df,cosine_sim=load_data()   #load the data and get the cosine sim matrix

#4.Generate recommendations 

    indices=pd.Series(df.index, index=df['title']).drop_duplicates()
#creating a pandas series with the title as the index and the row number as the value
#used to get the index of the movie by its title
#drop_duplicates() is used to not having problems with duplicate titles

    movies_list=df['title'].tolist() #all titles to create drop down menu
    selected_movie=st.selectbox('Select a movie or Tv show:',movies_list)

    if st.button('Get recommendations'):
       idx= indices[selected_movie]   #get the index of the movie that matches the title

       sim_scores=list(enumerate(cosine_sim[idx]))  #looks up the movie raw in the sim grid
    # enumerate() attaches the dataset index so we dont lose track of which movie the sim score belongs to
    # list() so its easy to sort

       sim_scores=sorted(sim_scores, key=lambda x:x[1], reverse=True)  #sort the tuple based on sim scores
    # reverse=True to sort in descending order

       sim_scores=sim_scores[1:6]  #get the top 5 sim movies excluding the first one (the same movie)

       movie_indices=[i[0] for i in sim_scores]    #get the movie indices

       recommendations=df[['title','listed_in','type','director']].iloc[movie_indices]

       st.subheader(f"Because you liked *{selected_movie}*, we recommend:")
       st.table(recommendations)

#------------------------------- END OF TASK 1 ------------------------------

#---------------------------------- TASK 2 ----------------------------------

elif page=='Task 2: Content prediction':
   st.title('Content Prediction Model')
   st.write('Machine learning models predicting if a title is a Movie or Tv show.')

#1.Loading the data

   @st.cache_resource        #bc ML models are heavy
   def train_models():
      df=pd.read_csv('Dataset.csv')

      df['has_director']=df['director'].apply(lambda x:0 if pd.isna(x) else 1)    
#create a column that indicates if the title has a director or not(tv shows mostly dont have directors)
    
      df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
      df['duration_num'] = df['duration_num'].fillna(0)

      
      df['rating']=df['rating'].fillna('Unknown')
      df['country']= df['country'].fillna('Unknown')

      X=df[['release_year','rating','country','has_director','duration_num']]
      y=df['type']

#2.Encoding
     
      le_rating=LabelEncoder()
      le_country=LabelEncoder()
      le_target=LabelEncoder()

      X_encoded=X.copy()
      X_encoded['rating']=le_rating.fit_transform(X['rating'])
      X_encoded['country']=le_country.fit_transform(X['country'])
      y_encoded=le_target.fit_transform(y)     #translates movies to 0 and tv shows to 1

#3.Spliting and training
     
      X_train,X_test,y_train,y_test=train_test_split(X_encoded,y_encoded,test_size=0.2,random_state=42)

      rf=RandomForestClassifier(random_state=42)
      rf.fit(X_train,y_train)
      lr=LogisticRegression(max_iter=1000, random_state=42)
      lr.fit(X_train,y_train)

#4.Evaluating the models

      rf_acc=accuracy_score(y_test,rf.predict(X_test))
      lr_acc=accuracy_score(y_test,lr.predict(X_test))
      return rf,lr,rf_acc,lr_acc,le_target

   rf_model,lr_model,rf_acc,lr_acc,le_target=train_models()

   st.header('1.Model Evalutation')
   col1,col2=st.columns(2)       #place metrics side by side on the page
   with col1:
      st.metric(label='Random Forest Accuracy', value=f"{rf_acc:.2%}")
   with col2:
      st.metric(label='Logistic Regression Accuracy', value=f"{lr_acc:.2%}")

   st.success('The Random Forest classifier performs better at distinguishing between Movies and TV Shows')
   st.divider()

   st.header('2.Test the model')
   st.write('Adjust the features below to see what the Random Forest model predicts.')

   input_year=st.slider('Release Year', 1925,2024,2020)
   input_director=st.radio('Has Director?',['Yes', 'No'])
   input_duration=st.slider('Duration (Length in Minutes OR Number of Seasons)',1,300,90)
#gets inputs from the user

   has_dir_value=1 if input_director=='Yes' else 0     #translates the user input to a numerical value
   sample_data=[[input_year,8,80,has_dir_value,input_duration]]    
#hardcode 8 and 80 are the encoded values for rating and country to keep it simple
   
   if st.button('Predict'):
      prediction_encoded=rf_model.predict(sample_data)
      prediction_text=le_target.inverse_transform(prediction_encoded)[0]    
#translates the numerical value back to the label
      
      if prediction_text=='Movie':
         st.info(f'The model predicts that this title is a **{prediction_text}**')
      else:
         st.info(f'The model predicts that this title is a **{prediction_text}**')
      
   
#------------------------------- END OF TASK 2 ------------------------------

#---------------------------------- TASK 3 ----------------------------------

elif page=='Task 3: Rating Classification':
   st.title('Audience Rating Classification')
   st.write('Predicts if a title is for Kids, Teens or Adults based on its genres, duration and type.')

   @st.cache_resource
   def train_task3_models():
      df=pd.read_csv('Dataset.csv')

#1.Analyze rating categories

      rating_map = {
    'TV-Y': 'Kids', 'TV-Y7': 'Kids', 'TV-G': 'Kids', 'G': 'Kids', 'TV-Y7-FV': 'Kids',
    'TV-PG': 'Teens', 'PG': 'Teens', 'PG-13': 'Teens', 'TV-14': 'Teens',
    'TV-MA': 'Adults', 'R': 'Adults', 'NC-17': 'Adults', 'NR': 'Adults', 'UR': 'Adults'}
#maping the 14 ratings into 3 categories to help train the model

      df['target_rating']=df['rating'].map(rating_map)   #new target column
      df=df.dropna(subset=['target_rating'])   #remove rows with blank rating to avoid confusion

#2.Prepare datasets

      df['duration_num']=df['duration'].str.extract(r'(\d+)').astype(float).fillna(0)
      #extract just the numbers

      df['type_num']=df['type'].apply(lambda x:1 if x=='Movie' else 0)

      cv= CountVectorizer(stop_words='english',max_features=500)
#to find the top 500 most occurring genres in the dataset

      genres_matrix=cv.fit_transform(df['listed_in']).toarray()
      genre_names=cv.get_feature_names_out()
      genres_df=pd.DataFrame(genres_matrix,columns=genre_names)
#creates a grid where each of the 50 genres becomes a column (1 if the movie has the genre, 0 if no)

      df=df.reset_index(drop=True)
      X=pd.concat([df[['duration_num','type_num']],genres_df],axis=1)
#combine the duration/ type numbers with the 50 new genre columns to create one massive feature table (X)

      le_target=LabelEncoder()
      y=le_target.fit_transform(df['target_rating'])    #kids=0 teens=1 adults=2

      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#3.Train classification models
   
      dt=DecisionTreeClassifier(random_state=42)   #creates a flowchart of yes/no questions
      rf=RandomForestClassifier(random_state=42)   #creates 100 different decision trees and their avg
      dt.fit(X_train, y_train)
      rf.fit(X_train, y_train)

#4.Optimize performance

      param_grid={'max_depth':[10,20],'n_estimators':[50,100]}
#settings we want to test(dic)
#max depth limits how deep the trees can grow, n estimators is how many trees to build

      grid_search=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=3)
#gridsearchcv trains a new random forest for every single combination of settings in the param_grid above
#cv=3 means it double checks its work 3 times for every combination to ensure it isnt just getting lucky
      
      grid_search.fit(X_train,y_train)
      best_rf=grid_search.best_estimator_
#save the ultimate winning model

#5.Evaluate

      dt_acc=accuracy_score(y_test, dt.predict(X_test))
      rf_acc=accuracy_score(y_test, rf.predict(X_test))
      best_rf_acc=accuracy_score(y_test, best_rf.predict(X_test))

      return dt,rf,best_rf,dt_acc,rf_acc,best_rf_acc,le_target,cv,X.columns

   dt,rf,best_rf,dt_acc,rf_acc,best_rf_acc,le_target,cv,feature_cols=train_task3_models()

   st.header('1.Model Comparison and Optimization')
   st.write('We tested three models.GridSearchCV hyperparameter tuning found the optimal settings.')
   col1,col2,col3=st.columns(3)
   with col1:
      st.metric(label='Decision tree',value=f"{dt_acc*100:.2f}%")
   with col2:
      st.metric(label='Basic random forest',value=f"{rf_acc*100:.2f}%")
   with col3:
      st.metric(label='Optimized random forest',value=f"{best_rf_acc*100:.2f}%")
   st.divider()

   st.header('2.Predict Audience Rating')
   st.write('Select the attributes of a show/movie to see what audience the optimized model recommends.')

   input_type=st.radio("Content Type:",["Movie", "TV Show"])
   input_duration=st.slider("Duration (Minutes or Seasons):",1,200,90)
   
   available_genres=cv.get_feature_names_out()
   selected_genres=st.multiselect('Select genres:',available_genres,default=['comedies'])

   if st.button('Predict'):
      type_num=1 if input_type=='Movie' else 0
      user_input_data={'duration_num':input_duration,'type_num':type_num}

      for genre in available_genres:
         user_input_data[genre]=1 if genre in selected_genres else 0

      input_df=pd.DataFrame([user_input_data],columns=feature_cols)
      pred=best_rf.predict(input_df)
      pred_text=le_target.inverse_transform(pred)[0]

      st.success(f"**Predicted audience:** {pred_text}")


#------------------------------- END OF TASK 3 ------------------------------

#---------------------------------- TASK 4 ----------------------------------
elif page=='Task 4: Content Segmentation':
   st.title('Netflix Content Segmentation')
   st.write('Grouping Netflix titles into meaningful clusters using Unsupervised Machine Learning.')

   @st.cache_resource
   def run_clustering():
      df=pd.read_csv('Dataset.csv')

#1.Prepare features

      df['duration_num']=df['duration'].str.extract(r'(\d+)').astype(float).fillna(0)
#extract numeric features only for clustering (k-means measure phy distance so everything must be numbers)

      df['type_num']=df['type'].apply(lambda x:1 if x=='Movie' else 0)
      df['release_year']=df['release_year'].fillna(df['release_year'].mode()[0])

      cv=CountVectorizer(stop_words='english',max_features=10)
      genres_matrix=cv.fit_transform(df['listed_in'].fillna(''))
      genres_df=pd.DataFrame(genres_matrix.toarray(),columns=cv.get_feature_names_out())
#turn the top 10 most common genres to binary columns 0 or 1

      X=pd.concat([df[['release_year','duration_num','type_num']],genres_df],axis=1)
#combine the num features and genre columns in the final dataset X
     
      scaler=StandardScaler()
      X_scaled=scaler.fit_transform(X)
#shrinks large numbers and expands small numbers so they are equal for the algorithm

#2.clustering

      kmeans=KMeans(n_clusters=4,random_state=42,n_init='auto')
#initialize and ask to find 4 distinct groups in the data

#3.Identify content groups
      df['Cluster']=kmeans.fit_predict(X_scaled)

#4.visualize clusters

      pca=PCA(n_components=2)   
      pca_components=pca.fit_transform(X_scaled)
#pca compresses 13 colums to X and Y , just 2 coordinates

      df['PCA_X']=pca_components[:,0]   
      df['PCA_Y']=pca_components[:,1]
#saves the new coordinates in the dataframe
     
      df['Cluster_Name']=df['Cluster'].map({0:'Groupe A', 1:'Groupe B', 2:'Groupe C', 3:'Groupe D'})
#renames the raw clusters 0,1,2,3 into readbale names
 
      return df,kmeans

   df_clustered,kmeans_model=run_clustering()
   st.header('1.Cluster Visulization (PCA)')
   st.write('We compressed the features into 2D space to visualize how K-Means grouped the content.')

   fig=px.scatter(
      df_clustered,
      x='PCA_X',
      y='PCA_Y',
      color='Cluster_Name',
      hover_data=['title','type','release_year','listed_in'],
      title='K-Means clusters of Netflix Content',
      color_discrete_sequence=px.colors.qualitative.Set1
   )
   st.plotly_chart(fig,use_container_width=True)
   st.divider()

#5.cluster characteristics
 
   st.header('2.Cluster Analysis')
   st.write('What do these groups actually represent? Here is the average profile of each cluster:')

   cluster_summary=df_clustered.groupby('Cluster_Name').agg(
      total_titles=('title','count'),
      most_common_type=('type',lambda x: x.mode()[0]),
      avg_release_year=('release_year',lambda x:int(x.mean())),
      avg_duration=('duration_num', lambda x:round(x.mean(),1))
   ).reset_index()

   st.table(cluster_summary)

#------------------------------- END OF TASK 4 ------------------------------