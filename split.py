#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys

from datetime import datetime
from collections import Counter
from subprocess import check_output

import matplotlib.pyplot as plt
import seaborn as sns


from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, from_unixtime

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator


sc = SparkContext('local')
spark = SparkSession(sc)

interaction = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header =True, schema='user_id INT,book_id INT, is_read INT,rating FLOAT, is_reviewed INT')
interaction.createOrReplaceTempView('interaction')
interaction = interaction.where('rating != 0')

unique_id = spark.sql('SELECT distinct user_id FROM interaction group by user_id having count(*) >=10 ')
fraction = float(sys.argv[1])
sample_id = unique_id.sample(fraction)

train_id, valid_id, test_id = sample_id.randomSplit(weights=[0.6,0.2,0.2])
#print(train_id.count(), valid_id.count(),test_id.count())

data_all = interaction.join(sample_id,on='user_id',how='inner').select('user_id','book_id','rating')
train = interaction.join(train_id,on='user_id',how='inner')
valid_all = interaction.join(valid_id,on='user_id',how='inner')
test_all = interaction.join(test_id,on='user_id',how='inner')
 
window = Window.partitionBy('user_id').orderBy('book_id') 
valid_all = valid_all.select('user_id','book_id','rating',F.row_number().over(window).alias("row_number"))
test_all = test_all.select('user_id','book_id','rating',F.row_number().over(window).alias("row_number"))

valid_train = valid_all.filter(valid_all.row_number % 2 ==1).drop('row_number')
valid = valid_all.filter(valid_all.row_number % 2 ==0).drop('row_number')

test_train = test_all.filter(test_all.row_number % 2 ==1).drop('row_number')
test = test_all.filter(test_all.row_number % 2 ==0).drop('row_number')

train_all = train.select('user_id','book_id','rating').union(valid_train).union(test_train)

valid_test = valid_train.union(valid).union(test_train).union(test)

valid_test.write.parquet('hdfs:/user/cl5293/recommend/{}_valid_test.parquet'.format(fraction*100))
data_all.write.parquet('hdfs:/user/cl5293/recommend/{}_data_all.parquet'.format(fraction*100))
train_all.write.parquet('hdfs:/user/cl5293/recommend/{}_train.parquet'.format(fraction*100))
valid.write.parquet('hdfs:/user/cl5293/recommend/{}_valid.parquet'.format(fraction*100))
test.write.parquet('hdfs:/user/cl5293/recommend/{}_test.parquet'.format(fraction*100))
