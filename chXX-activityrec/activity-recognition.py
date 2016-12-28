import os
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

PATH = '/Users/juliet/data/PAMAP2_Dataset/Protocol'

SENSOR_FIELDS = ['temp', 'acc-x-16', 'acc-y-16', 'acc-z-16',
                 'acc-x-6', 'acc-y-6', 'acc-z-6',
                 'gy-x', 'gy-y', 'gy-z',
                 'mag-x', 'mag-y', 'mag-z',
                 'orient-1', 'orient-2', 'orient-3', 'orient-4']

SENSOR_LOCATIONS = ['hand', 'chest', 'ankle']

ALL_SENSOR_FIELDS = ['-'.join([loc, name]) for loc in SENSOR_LOCATIONS for
                              name in SENSOR_FIELDS]

SCHEMA = StructType([StructField('subject_id', DoubleType(), True),
                     StructField('ts', DoubleType(), True),
		             StructField('activity_id', DoubleType(), True),
                     StructField('hr', DoubleType(), True)] +
                     [StructField(fieldname, DoubleType(), True) for
                     fieldname in  ALL_SENSOR_FIELDS])

def process_text_line(line, subject_id):
    elems = line.split(' ')
    tuple_line = tuple(float(elem) for elem in [subject_id] + elems)
    # 54 columns for original file and subject id
    assert len(tuple_line) == 55
    return tuple_line
    

def process_single_file(file_name):
    txt_lines = sc.textFile(os.path.join(PATH, file_name))
    subject_id = float(file_name[9:10])
    # map and create a bunch of tuples of observations
    observations = txt_lines.map(lambda line: process_text_line(line,
                                                                subject_id))
    return observations


spark = SparkSession.builder \
    .appName('PAMAP2 Activity Recognition') \
    .getOrCreate()
sc = spark.sparkContext

# Get all file names in data dir
files = [os.path.split(file)[1] for file in os.listdir(PATH)]
obs_rdds = [process_single_file(file_name) for file_name in files]
obs_rdd = sc.union(obs_rdds)
df = spark.createDataFrame(obs_rdd, SCHEMA)

# The orientation measurements are invalid, so we should drop them.
# The 6G accelerometers readings get saturated for some readings, so
# we will drop them in favor of the 16G sensors.
cols_to_drop = [field for field in ALL_SENSOR_FIELDS if ("orient" in field)
                or ("-6" in field)]
# Remove orientation columns because they are invalid
new = df.select([col for col in df.columns if col not in cols_to_drop])

mean_vals_df = new.groupBy(new.subject_id, new.activity_id).mean()
max_vals_df = new.groupBy(new.subject_id, new.activity_id).max()
min_vals_df = new.groupBy(new.subject_id, new.activity_id).min()

mean_max_jnd = mean_vals_df.join(max_vals_df,
                  (mean_vals_df.subject_id == max_vals_df.subject_id)
                & (mean_vals_df.activity_id == max_vals_df.activity_id))
mean_max_min_jnd = mean_max_jnd.join(min_vals_df,
                  (mean_max_jnd.subject_id == min_vals_df.subject_id)
                & (mean_max_jnd.activity_id == min_vals_df.activity_id))

# trigger execution and see if this worked at all
print(mean_max_min_jnd.take(1))

# we have some stats on each of these sensors now, so lets see if they are
# at all useful in predicting the type of activity:
# now each row in our df correspond to an label- featurevector pair. it is
# tidy data from the perspective of trying to attach labels to this tyoe of
# info.

# feature vector column maker
# figure out how to easily get all the numeirc columsn together
assembler = VectorAssembler(
    inputCols = ['sytuffffff1', 'stuffff2'],
    outputCol = 'features')

# string indexer for label generation
plan_indexer = StringIndexer(inputCol = 'activity-id',
                             outputCol = 'activity-type')



# toss some algorithm at it, random forests?
# Reg trees?



# Do feature generation
# https://issues.apache.org/jira/browse/SPARK-10915
# for each person and activity, create a dense vector.
# since pyspark does not have UDAFs we must do a 'collectlist'
# Either a numpy array or a list will be interpretted as a dense vector
vectorized_series = new.groupBy(new.subject_id, new.activity_id).agg(
    F.collect_list("hand-temp"))

# groupbys are expensive, cache this
vectorized_series.cache()

# summary stats on an activity
# <subjct id, activity id, {mean, min, max, 75%, 25%}x{hr, acc}>

# Get frequency space representation using DCT
#<subject id, activity id, DCT{gy-?, mag-?}>
# Probably first need to order list.

print(vectorized_series.take(1))
