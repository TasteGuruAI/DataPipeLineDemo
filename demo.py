from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import udf, array
from pyspark.sql.types import StringType
from textblob import TextBlob
import nltk
from pyspark.sql.functions import rand
import pickle
from itertools import combinations
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
import re
from operator import add, index
from functools import reduce

# load spacy and ner model
import spacy

nlp = spacy.load("NER_model")

def ner_extraction(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.ent_id_ == "":
            entities.append(ent.text)
        else:
            entities.append(ent.ent_id_)
    if entities == []:
        return ["empty"]
    else:
        return list(set(entities))


def transform_number(x):
    try:
        if "K" in str(x):
            return int(float(x[:-1]) * 1000)
        if "M" in str(x):
            return int(float(x[:-1]) * 1000000)
        else:
            return int(float(x))
    except:
        return int(0)


spark = SparkSession.builder.appName("DataTransformation").getOrCreate()

# get all the df in the df directory.
df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("./data2/*.csv")

df = df.orderBy(rand())
# df.show()
# filter the twitter
df2 = df.filter(df.Timestamp.isNotNull())
df2 = df2.limit(500)
# convert timestamp to the right format
timeStampPreCleaning = udf(
    lambda x: str(x) + " 2020" if len(x) < 8 else x.replace(",", ""), StringType()
)
df2 = df2.withColumn("Timestamp", timeStampPreCleaning("Timestamp"))

# StirngToDateType
df3 = df2.withColumn("TimeStampDateType", F.to_date(F.col("Timestamp"), "MMM dd yyyy"))

# drop null value rows which timestamp columns is not in the standard format.
df3 = df3.filter(df3.TimeStampDateType.isNotNull())
df3 = df3.withColumn("Year", F.year(df3.TimeStampDateType))
df3 = df3.withColumn("Month", F.month(df3.TimeStampDateType))
df3 = df3.withColumn("Qurter", F.quarter(df3.TimeStampDateType))
# fill null with 0 and convert unit to the right numbers.
cols = ["Comments", "Likes", "Retweets"]

df3 = df3.fillna("0", subset=cols)

# apply the transform_number udf
transformNumber = udf(lambda z: transform_number(z), T.IntegerType())
df3 = df3.withColumn("Comments", transformNumber("Comments"))
df3 = df3.withColumn("Likes", transformNumber("Likes"))
df3 = df3.withColumn("Retweets", transformNumber("Retweets"))
### check
logNormal = udf(lambda x: int(round(np.log2(x + 1)))+1, T.IntegerType())
df3 = df3.withColumn("Likes_log", logNormal("Likes"))
df3 = df3.withColumn("Retweets_log", logNormal("Retweets"))
df3 = df3.filter(df3.Likes_log.isNotNull())
df3 = df3.filter(df3.Retweets_log.isNotNull())

# get the keywords of queries used for scrapping tweets.
def extractkeyword(url):
    try:
        result = (
            re.search("searchq=(.+) until", url.replace("?", "").replace("%20", " "))
            .group(1)
            .replace(" lang%3Aen", "")
            .strip()
        )
        return result
    except:
        return None




extractKeywordFromQueries = udf(lambda x: extractkeyword(x))
df3 = df3.filter(df3.Page_URL.isNotNull())
df3 = df3.withColumn("Keyword", extractKeywordFromQueries("Page_URL"))
df3 = df3.filter(df3.Keyword.isNotNull())

# keywords = df.select("Keyword").rdd.flatMap(lambda x: x).collect()
# with open("keywords.pickle", "wb") as handle:
#     pickle.dump(keywords, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(set(keywords))
# df3.select(*cols).show()
SODA = ["fizzy drink", "soda", "sparkling water"]
TONIC = ["tonic"]
GINGERALE = ["ginger ale"]


def getCategory2(keyword):
    SODA = ["fizzy drink", "soda", "sparkling water"]
    TONIC = ["tonic"]
    GINGERALE = ["ginger ale", "coke", "pop"]
    if keyword in SODA:
        return "soda"
    if keyword in TONIC:
        return "tonic"
    if keyword in GINGERALE:
        return "ginger ale"


keywordToCategory2 = udf(lambda x: getCategory2(x), StringType())
df3 = df3.withColumn("Category2", keywordToCategory2("Keyword"))
f3 = df3.filter(df3.Category2.isNotNull())
# NER Model
# could be empty list,
nerExtraction = udf(lambda z: ner_extraction(z), T.ArrayType(StringType()))
# df3 = df3.fillna('None')
# TODO: fileter the text contain replying to @ before apply nerExtraction
df3 = df3.withColumn("All_phrases", nerExtraction("Text"))
df3 = df3.filter(df3.All_phrases.isNotNull())

def checkempty(phrasesList):
    if phrasesList == ['empty']:
        return int(1)
    else:
        return int(0)
    
checkEmpty = udf(lambda x: checkempty(x), T.IntegerType())

df3 = df3.withColumn('CheckEmpty',checkEmpty('All_phrases'))
df3 = df3.filter(df3.CheckEmpty.isNotNull())
df3 = df3.filter(df3.CheckEmpty == int(0))
df3 = df3.filter(df3.CheckEmpty.isNotNull())
col = ['Text']
df3 = df3.select(*col)
df3.toPandas().to_csv('Filtered_Text.csv',index=False)

# calculate sentiment
#------------------Waiting for Priscilla to Modify--------------------------#

# getSentiment = udf(lambda z: TextBlob(z).polarity, T.FloatType())
# df3 = df3.withColumn("Sentiment", getSentiment("Text"))






#------------------Waiting for Priscilla to Modify--------------------------#








# ### Get phrases frequency
# weighted_phrases_calculate = udf(
#     lambda x, y: y * (int(x) + 1), T.ArrayType(StringType())
# )


# df3 = df3.withColumn(
#     "Weighted_phrases", weighted_phrases_calculate("Retweets_log", "All_phrases")
# )


# # cols = ['Sentiment','All_phrases','Retweets_log','Weighted_phrases','Year','Month','Keyword']
# cols = ["Weighted_phrases", "Year", "Month", "Keyword", "Category2"]

# # cols = ["Weighted_phrases", "Year", "Month", "Keyword"]
# # cols = ['Sentiment','All_phrases','Retweets_log']
# # df3.select(*cols).show()
# pairs = df3.rdd.map(
#     lambda row: ((row.Year, row.Month, row.Category2), row.Weighted_phrases)
# )
# # print(pairs.collect())

# groupByMonth = pairs.groupByKey()


# def merge(month, phrasesList):
#     mergedPhraseList = []
#     for phrases in phrasesList:
#         mergedPhraseList.extend(phrases)
#     return (month, nltk.FreqDist(mergedPhraseList).most_common())


# Frequency_mergeByMonth = groupByMonth.map(lambda x: merge(x[0], x[1]))
# # print(Frequency_mergeByMonth.collect())
# df_freqmonth = Frequency_mergeByMonth.toDF(["Month", "Nested_List"])
# # df_freqmonth.show()
# df_freqmonth = df_freqmonth.select(
#     df_freqmonth.Month, F.explode(df_freqmonth.Nested_List)
# )
# Monthstring = udf(lambda x: "Frequency_" + str(x[0]) + "-" + str(x[1]), T.StringType())
# getCategory2Column = udf(lambda x: str(x[2]), T.StringType())

# df_freqmonth = df_freqmonth.withColumn('Category2',getCategory2Column("Month"))
# df_freqmonth = df_freqmonth.withColumn("Month", Monthstring("Month"))
# # df_freqmonth.show(50)
# getTopic = udf(lambda x: x[0], T.StringType())
# getFreq = udf(lambda x: x[1], T.IntegerType())
# df_freqmonth = df_freqmonth.withColumn("Topic", getTopic("col"))
# df_freqmonth = df_freqmonth.withColumn("Frequency", getFreq("col"))
# cols = ["Category2","Month", "Topic", "Frequency"]
# df_freqmonth = df_freqmonth.select(*cols)
# # # df_freqmonth.show()
# df_freqmonth = df_freqmonth.groupby(["Topic","Category2"]).pivot("Month").max("Frequency").fillna(0)
# df_freqmonth = df_freqmonth.withColumn('Category1',F.lit('Beverage'))
# df_freqmonth = df_freqmonth.filter(df_freqmonth.Topic!='empty')
# df_freqmonth.toPandas().to_csv('Frequency_monthly_demo.csv',index=False)



# # ----------------monthly sentiment 1d------------------_#
# """
# For each sentences get weighted sentiments| sentences | like 
# For each token in sentence => [(token,sentiments/likes)]

# groupby month, =>(month([token,sentiments/likes]))
# For each token get the sum of the sentiments and sum of the likes.
# """

# weighted_phrases_calculate = udf(lambda x, y: y * (int(x) + 1), T.FloatType())

# # get the weighted sentiments for each tweets.
# df3 = df3.withColumn(
#     "Weighted_Sentiment", weighted_phrases_calculate("Likes_log", "Sentiment")
# )


# sentiment_pairs = df3.rdd.map(
#     lambda row: (
#         (row.Year, row.Month,row.Category2),
#         (row.All_phrases, row.Weighted_Sentiment, row.Likes_log),
#     )
# )


# sentiment_groupByMonth = sentiment_pairs.groupByKey()
# '''
# last 3 month apple show 1 year ago sentiment no apple/ 


# '''


# def sentiment_process(month, monthIter):
#     """
#     Input =>
    
#     phrases : [string]
#     Weighted_Sentiment: Float
#     Likes : Int 
#     Output =>
#     ([(Vocabulary,Vocabulary_sentiment/vocabulary_like)])
    
#     """
#     phrases_list = []
#     vocabulary_sentiment = {}
#     vocabulary_likes = {}
#     # time to go home to cock

#     for phrases, Weighted_Sentiment, Likes in monthIter:
#         for phrase in phrases:
#             vocabulary_sentiment.setdefault(phrase, 0)
#             vocabulary_likes.setdefault(phrase, 0)
#             vocabulary_sentiment[phrase] += Weighted_Sentiment
#             vocabulary_likes[phrase] += Likes
#     return (
#         month,
#         [
#             (key, vocabulary_sentiment[key] / (vocabulary_likes[key] + 1))
#             for key in vocabulary_sentiment.keys()
#         ],
#     )


# processed_sentiment_groupByMonth = sentiment_groupByMonth.map(
#     lambda monthTuple: sentiment_process(monthTuple[0], monthTuple[1])
# )
# columName = ["Month", "Nested_List"]
# processed_sentiment_groupByMonth = processed_sentiment_groupByMonth.toDF(columName)
# df_sentiment = processed_sentiment_groupByMonth.select(
#     processed_sentiment_groupByMonth.Month,
#     F.explode(processed_sentiment_groupByMonth.Nested_List),
# )
# Monthstring = udf(lambda x: "Sentiment_" + str(x[0]) + "-" + str(x[1]), T.StringType())
# getCategory2Column = udf(lambda x: str(x[2]), T.StringType())
# df_sentiment = df_sentiment.withColumn('Category2',getCategory2Column("Month"))
# df_sentiment = df_sentiment.withColumn("Month", Monthstring("Month"))
# getTopic = udf(lambda x: x[0], T.StringType())
# getSentiments = udf(lambda x: x[1], T.FloatType())
# df_sentiment = df_sentiment.withColumn("Topic", getTopic("col"))
# df_sentiment = df_sentiment.withColumn("Sentiment", getSentiments("col"))
# cols = ['Category2',"Month", "Topic", "Sentiment"]
# df_sentiment = df_sentiment.select(*cols)
# df_sentiment = df_sentiment.groupby(["Topic",'Category2']).pivot("Month").max("Sentiment").fillna(0)
# df_sentiment = df_sentiment.withColumn('Category1',F.lit('Beverage'))
# df_sentiment = df_sentiment.filter(df_sentiment.Topic!='empty')
# df_sentiment.toPandas().to_csv('Sentiments_monthly_demo.csv',index=False)









# def combinationWithRetweets(combinationIter, Retweets_log):
#     print(Retweets_log)
#     result = []
#     for combination in combinationIter:
#         result.append((combination, Retweets_log + 1))
#     return result


# #


# def iterToList(iter):
#     result = []
#     for pair in iter:
#         result.append(pair)
#     return result


# sentiment_2d_pairs = df3.rdd.map(
#     lambda row: (
#         (row.Year, row.Month,row.Category2),
#         (
#             iterToList(combinations(row.All_phrases, 2)),
#             row.Weighted_Sentiment,
#             row.Likes_log,
#         ),
#     )
# )


# def sentiment_process(month, monthIter):
#     """
#     Input =>
    
#     phrase_pair : [(string,string)]
#     Weighted_Sentiment: Float
#     Likes : Int 
#     Output =>
#     ([(Vocabulary_pairs,Vocabulary_sentiment/vocabulary_like)])
    
#     """
#     phrases_list = []
#     vocabulary_sentiment = {}
#     vocabulary_likes = {}
#     # time to go home to cock 

#     for phrases_pair, Weighted_Sentiment, Likes in monthIter:
#         for phrase_pair in phrases_pair:
#             # phrase_pair =tuple(phrase_pair)
#             vocabulary_sentiment.setdefault(phrase_pair, 0)
#             vocabulary_likes.setdefault(phrase_pair, 0)
#             vocabulary_sentiment[phrase_pair] += Weighted_Sentiment
#             vocabulary_likes[phrase_pair] += Likes
#     return (
#         month,
#         [
#             (key, vocabulary_sentiment[key] / (vocabulary_likes[key] + 1))
#             for key in vocabulary_sentiment.keys()
#         ],
#     )



# sentiment2dGroupByMonth = sentiment_2d_pairs.groupByKey()

# processed_sentiment2dGroupByMonth = sentiment2dGroupByMonth.map(
#     lambda monthTuple: sentiment_process(monthTuple[0], monthTuple[1])
# )

# columName = ["Month", "Nested_List"]
# processed_sentiment2dGroupByMonth = processed_sentiment2dGroupByMonth.toDF(columName)

# df_sentiment_2d = processed_sentiment2dGroupByMonth.select(
#     processed_sentiment2dGroupByMonth.Month,
#     F.explode(processed_sentiment2dGroupByMonth.Nested_List),
# )

# Monthstring = udf(lambda x: "Sentiment_" + str(x[0]) + "-" + str(x[1]), T.StringType())
# getCategory2Column = udf(lambda x: str(x[2]), T.StringType())
# df_sentiment_2d = df_sentiment_2d.withColumn('Category2',getCategory2Column("Month"))
# df_sentiment_2d = df_sentiment_2d.withColumn("Month", Monthstring("Month"))
# getTopic = udf(lambda x: x[0][0], T.StringType())
# getTopic2 = udf(lambda x: x[0][1], T.StringType())
# getTopicPair = udf(lambda x: [x[0][1], x[0][0]])
# getSentiments = udf(lambda x: x[1], T.FloatType())
# df_sentiment_2d = df_sentiment_2d.withColumn("Topic", getTopic("col"))
# df_sentiment_2d = df_sentiment_2d.withColumn("Topic2", getTopic2("col"))
# df_sentiment_2d = df_sentiment_2d.withColumn("TopicPair", getTopicPair("col"))
# df_sentiment_2d = df_sentiment_2d.withColumn("Sentiment", getSentiments("col"))
# df_sentiment_2d = (
#     df_sentiment_2d.groupby('Category2',"Topic", "Topic2").pivot("Month").max("Sentiment").fillna(0)
# )
# df_sentiment_2d = df_sentiment_2d.withColumn('Category1',F.lit('Beverage'))
# # save df_sentiment_2d
# df_sentiment_2d = df_sentiment_2d.filter(df_sentiment_2d.Topic!='empty')
# df_sentiment_2d = df_sentiment_2d.filter(df_sentiment_2d.Topic2!='empty')
# df_sentiment_2d.toPandas().to_csv('Sentiment2D_monthly_demo.csv',index=False)

# # ---------------------------- SENTIMENT GROUP BY MONTHS DONE-------------------------#



# def combinationWithRetweets(Month,monthIter):
#     pairs_weight = {}
#     for phrases_pair,retweetslog in monthIter:
#         for phrase_pair in phrases_pair:
#             pairs_weight.setdefault(phrase_pair,1)
#             pairs_weight[phrase_pair] +=int(retweetslog)
#     return (Month,[(key,pairs_weight[key]) for key in pairs_weight.keys()])


# phrase_list_RDD = df3.rdd.map(
#     lambda row: ((row.Year, row.Month), (row.All_phrases, row.Retweets_log))
# )
# # result = phrase_list_RDD.map(lambda phrase_tuple:(combinations(phrase_tuple[0], 2),phrase_tuple[1]))\
# #                         .map(lambda combinations_tuple:combinationWithRetweets(combinations_tuple[0],combinations_tuple[1]))\
# #                         .flatMap(lambda x:x)\
# #                         .reduceByKey(lambda x,y:x+y).collect()

# freq_2d_pairs = df3.rdd.map(
#     lambda row: (
#         (row.Year, row.Month,row.Category2),
#         (iterToList(combinations(row.All_phrases, 2)), row.Retweets_log),
#     )
# )



# freq_2d_pairs_grouped = freq_2d_pairs.groupByKey()
# freq_2d_pairs_processed = freq_2d_pairs_grouped.map(lambda monthTuple: combinationWithRetweets(monthTuple[0],monthTuple[1]))
# columName = ["Month", "Nested_List"]
# processed_freq2dGroupByMonth = freq_2d_pairs_processed.toDF(columName)

# df_freq_month = processed_freq2dGroupByMonth.select(
#     processed_freq2dGroupByMonth.Month,
#     F.explode(processed_freq2dGroupByMonth.Nested_List),
# )
# Monthstring = udf(lambda x: "Frequency_" + str(x[0]) + "-" + str(x[1]), T.StringType())
# getCategory2Column = udf(lambda x: str(x[2]), T.StringType())

# df_freq_month = df_freq_month.withColumn('Category2',getCategory2Column("Month"))
# df_freq_month = df_freq_month.withColumn("Month", Monthstring("Month"))

# getTopic = udf(lambda x: x[0][0], T.StringType())
# getTopic2 = udf(lambda x: x[0][1], T.StringType())
# getTopicPair = udf(lambda x: [x[0][1], x[0][0]])
# getFrequency = udf(lambda x: x[1], T.IntegerType())
# df_freq_month = df_freq_month.withColumn("Topic", getTopic("col"))
# df_freq_month = df_freq_month.withColumn("Topic2", getTopic2("col"))
# df_freq_month = df_freq_month.withColumn("TopicPair", getTopicPair("col"))
# df_freq_month = df_freq_month.withColumn("Frequency", getFrequency("col"))
# df_freq_month = (
#     df_freq_month.groupby("Topic", "Topic2","Category2").pivot("Month").max("Frequency").fillna(0)
# )

# df_freq_month = df_freq_month.withColumn('Category1',F.lit('Beverage'))
# df_freq_month = df_freq_month.filter(df_freq_month.Topic!='empty')
# df_freq_month = df_freq_month.filter(df_freq_month.Topic2!='empty')
# df_freq_month.toPandas().to_csv('Frequency_2d_monthly_demo.csv',index=False)


