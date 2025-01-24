import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataPreparation:

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute les valeurs manquantes d'un DataFrame.
        
        - Colonnes numériques : Imputation par la médiane.
        - Colonnes catégorielles : Imputation par le mode.

        Paramètres :
        - df : pd.DataFrame : Le DataFrame à traiter.

        Retourne :
        - pd.DataFrame : DataFrame avec les valeurs manquantes imputées.
        """
        for column in df.columns:
            
            if df[column].isna().sum() > 0:  
                
                if pd.api.types.is_numeric_dtype(df[column]):  
                    df[column] = df[column].fillna(df[column].median()) 
                    
                elif pd.api.types.is_object_dtype(df[column]): 
                    df[column] = df[column].fillna(df[column].mode()[0]) 
                    
        return df
    

    @staticmethod
    def taux_na(df: pd.DataFrame) -> pd.Series:
        """
        Calcule le pourcentage de valeurs manquantes pour chaque colonne d'un DataFrame.

        Paramètres :
        - df : pd.DataFrame : Le DataFrame à analyser.

        Retourne :
        - pd.Series : Série contenant les pourcentages de valeurs manquantes par colonne, triés par ordre décroissant.
        """
        taux = (df.isna().sum() / len(df)) * 100 
        return taux[taux > 0].sort_values(ascending=False)  
    
    
    
    @staticmethod
    def preprocessing(df: pd.DataFrame, idcol: str = 'ID', dropcol: list = []):
        """
        Pré-traite le DataFrame en supprimant les colonnes spécifiées, en encodant les colonnes catégorielles,
        et en séparant l'ID.

        Paramètres :
        - df : pd.DataFrame : Le DataFrame à traiter.
        - idcol : str : Le nom de la colonne représentant l'ID (par défaut 'ID').
        - dropcol : list : Liste des colonnes à supprimer du DataFrame (par défaut liste vide).

        Retourne :
        - pd.DataFrame : Le DataFrame traité avec les colonnes supprimées et les valeurs encodées.
        - pd.Series : La colonne des ID séparée.
        - LabelEncoder : L'instance du `LabelEncoder` pour pouvoir l'utiliser ultérieurement.
        """
        
        lbEncoder = LabelEncoder()

        id_series = df[idcol]
        df = df.drop([idcol] + dropcol, axis=1)
        
        # Encoder les colonnes catégorielles
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = lbEncoder.fit_transform(df[column])
        
        return df, id_series, lbEncoder
