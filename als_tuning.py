#!/usr/bin/env python
import math
import datetime
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from pyspark.sql.functions import when

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

def main(spark):
    print(spark.sparkContext.getConf().getAll())


    train = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_train.parquet')
    test = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_valid.parquet')
    df = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_valid_test.parquet')

    perUserActualItemsDF = df.withColumn('rating', when(df.rating<3, None).otherwise(df.rating))\
        .groupBy("user_id").agg(F.collect_list("book_id").alias("Actual"))



    for regParam in [0.05,0.1,0.5,1]:
        als = ALS(rank=150, maxIter=10, regParam=regParam, alpha = 1,  
        userCol='user_id',
        itemCol='book_id',
        ratingCol='rating',
        coldStartStrategy="drop"
        )

        start_train = datetime.datetime.now()

        model = als.fit(train)

        end_train = datetime.datetime.now()
        
        user_subset_recs = model.recommendForUserSubset(test.select('user_id'), 500)
        perUserPredictItemsDF = (user_subset_recs.groupBy('user_id')
        .agg(F.expr('collect_list(recommendations.book_id)[0] as books')))

        end_subset = datetime.datetime.now()

        perUserItemsRDD = perUserPredictItemsDF.join(perUserActualItemsDF, 'user_id','left') \
        .rdd \
        .map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(perUserItemsRDD)

        print('rank: ',rank, 'regParam: ', regParam, 'alpha: ', alpha)
        print('mean Average Precision: ',rankingMetrics.meanAveragePrecision)
        print('precision At (500): ',rankingMetrics.precisionAt(500))
        print('ndcg At (500): ',rankingMetrics.ndcgAt(500),'\n')

        start_transform = datetime.datetime.now()
        predictions = model.transform(test)
        end_transform = datetime.datetime.now()

        windowSpec = Window.partitionBy('user_id').orderBy(F.col('prediction').desc())
        perUserPredictItemsDF = (predictions
        .select('user_id', 'book_id', 'prediction', F.rank().over(windowSpec).alias('rank'))
        .where('rank <= 500')
        .groupBy('user_id')
        .agg(F.expr('collect_list(book_id) as books')))


        perUserItemsRDD = perUserPredictItemsDF.join(perUserActualItemsDF, 'user_id','left').repartition(1000) \
        .rdd \
        .map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(perUserItemsRDD)

        print('mean Average Precision: ',rankingMetrics.meanAveragePrecision)
        print('precision At (500): ',rankingMetrics.precisionAt(500))
        print('ndcg At (500): ',rankingMetrics.ndcgAt(500),'\n')

        print('training time: ', end_train - start_train)
        print('prediction (subset) time: ', end_subset - end_train)
        print('prediction (transform) time: ', end_transform - start_transform)


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
