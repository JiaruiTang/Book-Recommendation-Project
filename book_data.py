import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions

books = spark.read.json('hdfs:/user/jt3869/goodreads_books.json.gz')
books.createOrReplaceTempView('books')

table1 = spark.sql('SELECT book_id, average_rating, country_code, language_code, authors.author_id[0] as author_1, authors.author_id[1] as author_2, popular_shelves.name[0] as popular_shelf_1, popular_shelves.name[1] as popular_shelf_2 FROM books')
table1.write.csv('hdfs:/user/jt3869/books_info.csv')

table1.createOrReplaceTempView('book_info')

item_factor = spark.read.csv('hdfs:/user/jt3869/itemFactors_total.csv')
item_factor.createOrReplaceTempView('item_factor')

book_id_map = spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header=True)
book_id_map.createOrReplaceTempView('book_id_map')

item_factor_id = spark.sql('select * from item_factor left join book_id_map on item_factor._c0 = book_id_map.book_id_csv')
item_factor_id.createOrReplaceTempView('item_factor_id')

item_factor_id.write.csv('hdfs:/user/jt3869/item_factor_id.csv')

book_join = spark.sql('select * from item_factor_id left join book_info on item_factor_id.book_id = book_info.book_id')
book_join.createOrReplaceTempView('book_join')
book_join.write.csv('hdfs:/user/jt3869/book_join.csv')

book_join1 = book_join.drop('book_id')

book_join1.write.csv('hdfs:/user/jt3869/book_join1.csv')
book_join1 = book_join.drop('book_id')

book_join1.write.csv('hdfs:/user/jt3869/book_join1.csv')
book_join1.createOrReplaceTempView('book_join1')
