from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, format_number

# Create a SparkSession
spark = SparkSession.builder.appName("json_example").getOrCreate()

# Read the JSON file
data_frame = spark.read.option("multiline","true").json("books.json")

# Filter English books
df_english = data_frame.filter(data_frame.language == 'English')

# Calculate average number of pages per author
avg_pages = df_english.groupBy("author").agg(avg("pages").alias("avg_pages"))

# Format avg_pages to 2 decimal places
avg_pages = avg_pages.withColumn("avg_pages", format_number("avg_pages", 2))

# Count the number of rows
num_rows = avg_pages.count()

# Show all rows
avg_pages.show(num_rows)
