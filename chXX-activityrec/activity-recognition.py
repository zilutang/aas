import os
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import DoubleType

PATH = '/Users/juliet/data/PAMAP2_Dataset/Protocol'

SENSOR_FIELDS = ['temp', 'acc-x-16', 'acc-y-16', 'acc-z-16',
                 'acc-x-6', 'acc-y-6', 'acc-z-6',
                 'gy-x', 'gy-y', 'gy-z',
                 'mag-x', 'mag-y', 'mag-z',
                 'orient-1', 'orient-2', 'orient-3', 'orient-4']

SENSOR_LOCATIONS = ['hand', 'chest', 'ankle']

ALL_SENSOR_FIELDS = ['-'.join([loc, name]) for loc in SENSOR_LOCATIONS for
                              name in SENSOR_FIELDS]

SCHEMA = StructType([StructField('subjectid', DoubleType(), True),
                     StructField('ts', DoubleType(), True),
		             StructField('activityID', DoubleType(), True),
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
cols_to_drop = [field for field in ALL_SENSOR_FIELDS if "orient" in field]
assert len(cols_to_drop) == 12, "col len is {}".format(cols_to_drop)
print("XX{}".format(cols_to_drop))
new = df.select([col for col in df.columns if col not in cols_to_drop])
print(new.take(1))
