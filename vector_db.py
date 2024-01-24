from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

catalog_status = spark.readStream.format("mongodb").\
            option('spark.mongodb.connection.uri', MONGO_CONN).\
            option('spark.mongodb.database', "search").\
            option('spark.mongodb.collection', "catalog_myn").\
option('spark.mongodb.change.stream.publish.full.document.only','true').\
            option('spark.mongodb.aggregation.pipeline',[]).\
            option("forceDeleteTempCheckpointLocation", "true").load()

catalog_status = catalog_status.withColumn("discountedPrice", F.col("price") * F.col("pred_price"))
catalog_status = catalog_status.withColumn("atp", (F.col("atp").cast("boolean") & F.lit(1).cast("boolean")).cast("integer"))

catalog_status.withColumn("vec", get_vec("title"))

catalog_status = catalog_status.drop("_id")
catalog_status.writeStream.format("mongodb").\
            option('spark.mongodb.connection.uri', MONGO_CONN).\
            option('spark.mongodb.database', "search").\
            option('spark.mongodb.collection', "catalog_final_myn").\
            option('spark.mongodb.operationType', "update").\
            option('spark.mongodb.idFieldList', "id").\
            option("forceDeleteTempCheckpointLocation", "true").\
            option("checkpointLocation", "/tmp/retail-atp-myn5/_checkpoint/").\
            outputMode("append").\
            start()

