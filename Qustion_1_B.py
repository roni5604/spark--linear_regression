from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Create a SparkSession
spark = SparkSession.builder.appName("json_example").getOrCreate()

# Read the JSON file
df = spark.read.option("multiline","true").json("books.json")

# Filter English books
df_english = df.filter(df.language == 'English')

# Calculate average number of pages per author
avg_pages = df_english.groupBy("author").agg(avg("pages").alias("avg_pages"))

# Count the number of rows
num_rows = avg_pages.count()

# Show all rows
avg_pages.show(num_rows)
