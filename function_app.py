import azure.functions as func
import logging
import azure.functions as func
import pickle   
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import random 
from enum import Enum
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from io import BytesIO
from azure.storage.blob import BlobClient
from enum import Enum

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Define an enum for functions
class Mode(Enum):
    MEAN = "mean"
    RANDOM = "random"
    LAST = "last"


def download_dataframe_from_blob(sas_url):
    blob_client = BlobClient.from_blob_url(sas_url)
    blob_bytes = blob_client.download_blob().readall()
    df = pd.read_pickle(BytesIO(blob_bytes))

    return df 

def content_based_filtering(userID, clicks, df_embeddings, n_recommendations,mode=Mode.LAST):
    #établir la liste des articles lus par l'utilisateur
    user_history = clicks[clicks.user_id == userID]['article_id'].tolist()
    embeddings = df_embeddings.to_numpy()

    #Supprimer les articles déjà lus de la matrice pour éviter de suggérer un article déjà lu
    filtered_matrix = np.delete(embeddings, user_history, axis=0)



    if mode==Mode.MEAN:
        user_history_df = clicks[clicks.user_id == userID]
        user_history_df.sort_values(by="article_id", inplace=True)
        filtered_df = df_embeddings.iloc[user_history_df["article_id"].to_list()]
        filtered_np = filtered_df.to_numpy()
        weights = user_history_df['session_size'].to_list()

        selected_article_vector = np.zeros((1,filtered_np.shape[1]))

        for i,j in zip(range(len(weights)),weights) :

            selected_article_vector = selected_article_vector +  filtered_np[i] * j

        selected_article_vector = selected_article_vector/sum(weights)

         #Calculer la matrice de similarité 
    
        similarity_matrix = cosine_similarity(selected_article_vector.reshape(1, -1),filtered_matrix)

        #Identifier les n articles les plus similaires 
        sorted_indices = np.argsort(similarity_matrix[0])[::-1]
        recommendations =  sorted_indices[:n_recommendations]

        print("Vous avez aimé ", ' '.join(map(str, user_history)), "? Nous vous suggérons ", ' '.join(map(str, recommendations)))
        
    elif mode==Mode.RANDOM:
        selected_article = random.choice(user_history)
        
        
        
        #Calculer la matrice de similarité 
        selected_article_vector = embeddings[selected_article]
        similarity_matrix = cosine_similarity(selected_article_vector.reshape(1, -1),filtered_matrix)

        #Identifier les n articles les plus similaires 
        sorted_indices = np.argsort(similarity_matrix[0])[::-1]
        recommendations =  sorted_indices[:n_recommendations]
        
        print("Vous avez aimé ", selected_article, "? Nous vous suggérons ", ' '.join(map(str, recommendations)))

    elif mode==Mode.LAST:
        selected_article = user_history[-1]

        
        #Calculer la matrice de similarité 
        selected_article_vector = embeddings[selected_article]
        similarity_matrix = cosine_similarity(selected_article_vector.reshape(1, -1),filtered_matrix)

        #Identifier les n articles les plus similaires 
        sorted_indices = np.argsort(similarity_matrix[0])[::-1]
        recommendations =  sorted_indices[:n_recommendations]
        
        print("Vous avez aimé ", selected_article, "? Nous vous suggérons ", ' '.join(map(str, recommendations)))

    return recommendations

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    embeddings =  download_dataframe_from_blob("https://recommendationcontent.blob.core.windows.net/embeddings/articles_embeddings.pkl?sp=r&st=2023-09-30T00:53:04Z&se=2023-09-30T08:53:04Z&sv=2022-11-02&sr=b&sig=fPhhFda4nfaTskk8nSGeWXUlIoBJ%2FpaKOXYRm3Nct34%3D")
    clicks = download_dataframe_from_blob("https://recommendationcontent.blob.core.windows.net/clicks/clicks.pkl?sp=r&st=2023-09-30T00:53:57Z&se=2023-09-30T08:53:57Z&sv=2022-11-02&sr=b&sig=PQjQtC9a3B5k7Lt8aptItHgzAQeV7KiXIDMJU4l2UV4%3D")
    userID = req.params.get('userId')
    
    name = req.params.get('name')
    if not userID:
        try:
            data = req.get_json()
            userID = data.get('userId')
        except ValueError:
            return func.HttpResponse(
                "Please provide a userId'",
                status_code=200
            )

    if userID:
        # récupération 5 articles recommandés
        userID = int(userID)
        reco = content_based_filtering(int(userID), clicks, embeddings, 5)

        # conversion liste en string
        result = (' '.join(str(elem)+"," for elem in reco))[:-1]

        #liste_reco = []
        #for r in reco:
        #    liste_reco.append(int(r))

        # retourne recommendations
        return func.HttpResponse(result, status_code=200)

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

    
