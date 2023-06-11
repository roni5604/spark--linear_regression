from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Create a SparkSession
spark = SparkSession.builder.appName("json_example").getOrCreate()

# Read the JSON file
# df = spark.read.json("books.json")
df = spark.read.option("multiline","true").json("books.json")


# Filter authors that start with 'F'
df_filtered = df.filter(df.author.startswith('F'))

# Select title, author and calculate years from the book's year to 2023
df_selected = df_filtered.select(col("title").alias("Book Name"), "author", (lit(2023) - df.year).alias("years_since_publication"))

# Count the number of rows
num_rows = df_selected.count()

# Show the data
df_selected.show(num_rows)
