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

from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns

sc = SparkContext('local')
spark = SparkSession(sc)

train = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_train.parquet')

als = ALS(rank=150, maxIter=10, regParam=0.005,
    userCol='user_id',
    itemCol='book_id',
    ratingCol='rating',
    coldStartStrategy="drop",
    nonnegative=True
    )
model = als.fit(train)

itemFactors = model.itemFactors
itemFactors=itemFactors.withColumnRenamed("id", "book_id")
item = itemFactors.select(itemFactors.columns+[(col("features")[x]).alias("feature"+str(x+1)) for x in range(0, 150)])
item = item.drop('features')

book = spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv',header=True)
genre = spark.read.json('hdfs:/user/cl5293/recommend/goodreads_book_genres_initial.json')
genre_id = genre.join(book,on='book_id',how='inner') 
genre_id = genre_id.drop('book_id')
genre_id=genre_id.withColumnRenamed("book_id_csv", "book_id")

genre_df =genre_id.select('book_id',F.expr('genres.children'),F.expr('genres.`comics, graphic`'), \
                       F.expr('genres.`fantasy, paranormal`'), \
                     F.expr('genres.fiction'),F.expr('genres.`history, historical fiction, biography`'), \
                      F.expr('genres.`mystery, thriller, crime`'), F.expr('genres.`non-fiction`'), \
                        F.expr('genres.poetry'),F.expr('genres.romance'),F.expr('genres.`young-adult`'))
genre_df = genre_df.fillna(0)
item_genre = item.join(genre_df,on='book_id',how='inner')
item_genre_sample = item_genre.sample(0.1)

genres = itemFactors_genre_sample[['children', 'comics, graphic',
       'fantasy, paranormal', 'fiction',
       'history, historical fiction, biography', 'mystery, thriller, crime',
       'non-fiction', 'poetry', 'romance', 'young-adult']]
genres['genre']=np.argmax(genres.values,axis=1)

features_embedded = TSNE(n_components=2).fit_transform(itemFactors_genre_sample.iloc[:,1:151].values)
df_embedded=pd.DataFrame(features_embedded)
df_embedded['genre']=genres_sample['genre']
df_embedded.columns = ['x','y','genre']

df_embedded_normalize=pd.DataFrame(preprocessing.normalize(df_embedded[['x','y']],axis=0))
df_embedded_normalize['genre']=genres['genre']
df_embedded_normalize.columns = ['x','y','genre']

ax = sns.scatterplot(x='x', y="y", hue='genre',data=df_embedded,legend="full")
ax.savefig("output.png")
