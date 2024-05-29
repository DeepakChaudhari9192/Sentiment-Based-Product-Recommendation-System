import pickle
import pandas as pd 
import numpy as np  

class ProdRecommender:
    root_model_path = "models/"
    sentiment_model = "best_sentiment_model.pkl"
    tfidf_vectorizer = "tfidf.pkl"
    best_recommender = "best_recommendation_model.pkl"
    clean_dataframe = "cleaned_dataframe.pkl"

    def __init__(self):
        self.sentiment_model = load(ProdRecommender.root_model_path + ProdRecommender.sentiment_model)
        self.tfidf_vectorizer = pd.read_pickle(ProdRecommender.root_model_path + ProdRecommender.tfidf_vectorizer)
        self.recommendation_model = load(ProdRecommender.root_model_path + ProdRecommender.best_recommender)
        self.cleaned_data = load(ProdRecommender.root_model_path + ProdRecommender.clean_dataframe)

        
    def product_recommendation (self, user_name):
    
        if user_name not in self.recommendation_model.index:
            print( 'The user {} does not exist. Please enter a valid user name'.format(user_name) )
        
        else:
            # Picking Top 20 product for the user from top 20 recommendation model
            
            top20_recommendated_product = list( self.recommendation_model.loc[user_name].sort_values(ascending = False)[:20].index )
            
            # get processed reviews for top 20 products
            
            top20_product_df = self.cleaned_data [ self.cleaned_data.name.isin( top20_recommendated_product )]
            
            # Converting the cleaned review into TFIDF vectorizer form so it can be easy to pass through logistic regression model
            
            X = self.tfidf_vectorizer.transform( top20_product_df['cleaned_review'].values.astype(str) )
            
            # Using the best sentiment model for sentiment prediction
            
            top20_product_df['predicted_sentiment'] = self.sentiment_model.predict(X)
            
            # Creating a new column which shows 1 for positive and 0 for Negative sentiment
            
            top20_product_df['positive_sentiment'] = top20_product_df['predicted_sentiment'].apply(lambda x: 1 if x=='Positive' else 0)
            
            # Creating a dataframe which stores a information related to sentiments
            
            final_df = top20_product_df.groupby( by = 'name' ).sum()
            final_df.columns = ['count_pos']
            
            # adding column to measure a total number of sentiments
            
            final_df['total_sentiments'] = top20_product_df.groupby(by='name')['predicted_sentiment'].count()
            
            # adding a column which measure a % of positive review out of total reviews
            
            final_df['positive_review_%'] = np.round( final_df['count_pos'] / final_df['total_sentiments']*100, 2 )
            
            # storing Top 5 recommendation based on positive review percentage.
            
            top_5_products = list(final_df.sort_values( by= 'positive_review_%', ascending=False)[:5].index)
            
            # Returning Top 5 product as a output
            
            return top_5_products
       

