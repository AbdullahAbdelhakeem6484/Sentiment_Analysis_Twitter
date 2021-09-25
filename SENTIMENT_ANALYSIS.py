#!/usr/bin/env python
# coding: utf-8

# In[1]:


################################################################################################################################
# Author        : - Abdullah Abdelhakeem                                                                                       #  
# Date          : - 24-9-2021                                                                                                  #  
# Version       : - v 0.0.1                                                                                                    #  
# Project       : - Real Time Streaming Sentiment Analysis (+ve , -ve , N) Twitter.                                                    #  
#                                                                                                                              # 
# Dependencies  : -   1- Tweepy            
#                     2- Apache Kafka      
#                     3- Apache Spark      
#                     4- kafka-python      
#                     5- pySpark           
#                     6- Delta Lake package                                                                                                         
#
#
#
#                                                                                                                              #
# Steps :                                                                                                                      # 
#        1- install Dependency                                                                                                 # 
#        2- Load Data                                                                                                          # 
#        3- Cleaning the Data                                                                                                  # 
#        4- PreProcessing the Data                                                                                             # 
#        5- Feature Selection                                                                                                  # 
#        6- Tokenize the data                                                                                               # 
#        7- Split the data 
#        8- Create an ML Pipeline
#        9- tarin the model(fit) and predict
#        10-Calculate Accuracy
#        11-save the model for deploying
#        12-load the model saved(check)
#                                                                                                                              #
###############################################################################################################################


# In[2]:


################################################################################################################################ 
#                                                                                                                              #
# Requirements :                                                                                                               # 
#        1- Tweepy                                                                                                             # 
#        2- Apache Kafka                                                                                                       # 
#        3- Apache Spark                                                                                                       # 
#        4- kafka-python                                                                                                       # 
#        5- pySpark                                                                                                            # 
#        6- Delta Lake package                                                                                                 # 
#                                                                                                                              # 
#                                                                                                                              #          
###############################################################################################################################


# In[3]:


'''
Reads 'Sentiment140' dataset, trains and saves the pipeline 
using SparkML
'''

import findspark
findspark.init()
import pyspark


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import udf
import re

from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.feature import StopWordsRemover, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pathlib import Path

from pyspark.sql.types import StringType, StructType, StructField, ArrayType
from pyspark.sql.functions import udf, from_json, col

from pyspark.ml import PipelineModel

import re
from datetime import datetime
from pathlib import Path

from tweepy import OAuthHandler, StreamListener
from tweepy import Stream, API
# from kafka import KafkaProducer

import json
from dateutil.parser import parse
import re


# In[4]:


# # Include information necessary for Twitter API authentication
# Developer Account Take 15 days
# access_token = 'ACCESS_TOKEN'
# access_token_secret = 'ACCESS_TOKEN_SECRET'
# consumer_key = 'CONSUMER_KEY'
# consumer_secret = 'CONSUMER_SECRET'


# In[ ]:





# In[5]:


import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').appName("RDDApp").getOrCreate()
spark = SparkSession.builder.getOrCreate()
sc =spark.sparkContext


# In[6]:


sqlcontext = SQLContext(sc)


# # Load The Data

# In[7]:


'''
Sentiment Analysis dataset
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet 
2 - the date of the tweet 
3 - the query . If there is no query, then this value is NO_QUERY.
4 - the user that tweeted 
5 - the text of the tweet 
'''

raw_data = sqlcontext     .read     .format('csv')     .options(header=False)     .load("train.csv")     .selectExpr("_c0 as sentiment","_c1 as id","_c2 as date ","_c3 as query","_c4 as user","_c5 as message")


# In[8]:


raw_data.show()


# In[9]:


raw_data.printSchema()


# In[10]:


df = sqlcontext     .read     .format('csv')     .options(header=False)     .load("train.csv")     .selectExpr("_c0 as sentiment", "_c5 as message")


# In[11]:


df.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Tokenize the data

# In[12]:


pre_process = udf(
    lambda x: re.sub(r'[^A-Za-z\n ]|(http\S+)|(www.\S+)', '', \
        x.lower().strip()).split(), ArrayType(StringType())
    )
df = df.withColumn("cleaned_data", pre_process(df.message)).dropna()


# In[13]:


df.show()


# In[14]:


# df.select('cleaned_data').toPandas()


# In[ ]:





# In[ ]:





# In[ ]:





# # Split the data into training and testing

# In[15]:


train, test = df.randomSplit([0.8,0.2],seed = 100)


# In[16]:


train.show()


# In[17]:


import pyspark
def spark_shape(self):
    return (self.count(), len(self.columns))
pyspark.sql.dataframe.DataFrame.shape = spark_shape


# In[18]:


print(train.shape())


# In[ ]:





# In[19]:


test.show()


# In[20]:


test.shape()


# # Create an ML Pipeline

# In[21]:


# Peforms TF-IDF calculation and Logistic Regression
remover = StopWordsRemover(inputCol="cleaned_data", outputCol="words")
vector_tf = CountVectorizer(inputCol="words", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features", minDocFreq=3)
label_indexer = StringIndexer(inputCol = "sentiment", outputCol = "label")
lr_model = LogisticRegression(maxIter=100)

pipeline = Pipeline(stages=[remover, vector_tf, idf, label_indexer, lr_model])


# # Fit the pipeline to the training dataframe

# In[22]:


trained_model = pipeline.fit(train)


# # Predicting the test dataframe 

# In[23]:


'''
The labels are labelled with positive (4) as 0.0 
negative (0) as 1.0
'''
prediction_df = trained_model.transform(test)
prediction_df.printSchema()


# In[24]:


prediction_df.select("message","features","label","prediction").show(100)


# In[25]:


prediction_df.select("features").show(1 , truncate=False)


# # Calculate Accuracy

# In[26]:


evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy = evaluator.evaluate(prediction_df)
print(accuracy)


# In[27]:


from pyspark.ml.evaluation import RegressionEvaluator
regEval = RegressionEvaluator(predictionCol ='prediction', labelCol='label',metricName='rmse')
regEval.evaluate(prediction_df)


# In[28]:


regEval_r2 = RegressionEvaluator(predictionCol ='prediction', labelCol='label',metricName='r2')
regEval_r2.evaluate(prediction_df)


# In[ ]:





# In[ ]:





# # Save the pipeline model

# In[29]:


# trained_model.save("fittedpipeline_Model")


# In[ ]:





# # Save the Model for Deploy

# In[30]:


trained_model.write()             .overwrite()              .save("ModelDir")


# # Load the Model Saved

# In[31]:


# read pickled model via pipeline api
from pyspark.ml.pipeline import PipelineModel
persistedModel = PipelineModel.load("ModelDir")


# In[32]:


# predict
predictionsDF = persistedModel.transform(test)


# In[33]:


predictionsDF.select("message","features","label","prediction").show(5)


# # Thank you! Abdullah Abdelhakeem
