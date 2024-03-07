#!/usr/bin/env python
import math
import datetime
from pyspark.sql import SparkSession

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from pyspark.sql.functions import when

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

def main(spark):
    print(spark.sparkContext.getConf().getAll())
    
    # Abstracting load of split data
    
    train = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_train.parquet')
    test = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_test.parquet')
    df = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_valid_test.parquet')

    als = ALS(
        userCol='user_id',
        itemCol='book_id',
        ratingCol='rating',
        coldStartStrategy="drop")
    model = als.fit(train)
    
    user_subset_recs = model.recommendForUserSubset(test.select('user_id'), 500)

    data=[]
    for user, items in user_subset_recs.collect():
        predict_items = [i.book_id for i in items]
        data.append((user,predict_items))
    perUserPredictItemsDF = spark.createDataFrame(data, schema=['user_id', 'book_id'])

    perUserActualItemsDF = df.withColumn('rating', when(df.rating<3, None).otherwise(df.rating))\
    .groupBy("user_id").agg(F.collect_list("book_id").alias("Actual"))
    
    perUserItemsRDD = perUserPredictItemsDF.join(perUserActualItemsDF, 'user_id','left') \
    .rdd \
    .map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(perUserItemsRDD)

    print('mean Average Precision: ',rankingMetrics.meanAveragePrecision)
    print('precision At (500): ',rankingMetrics.precisionAt(500))
    print('ndcg At (500): ',rankingMetrics.ndcgAt(500))


if __name__ == "__main__":
    memory = '5g'
    spark = (SparkSession.builder
             .appName('train_als')
             .master('local[*]')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main(spark)
    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    spark.stop()