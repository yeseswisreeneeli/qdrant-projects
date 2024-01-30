from product_search.data_loading import load_json
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from tqdm import tqdm

#instantiate qdrant client
print("[INFO] Client created...")
qdrant = QdrantClient("localhost", port=6333)

print("[INFO] Loading the dataset ... ")
dataset = load_json("data.json")

print("[INFO] Processing Data ...")
columns = ['type', 'name', 'description', 'cost', 'color']
data_to_embd = []
for i in range(len(dataset['makeupProducts'])):
    lst = [dataset['makeupProducts'][i][key] for key in columns]
    concated = ' '.join(lst)
    data_to_embd.append(concated)
    
print("[INFO] Data ready to feed into the model..")

print("[INFO] Loading encoder model...")
encoder = SentenceTransformer('thenlper/gte-small')

#creating data collection in qdrant
print("[INFO] Creating a data collection...")
qdrant.recreate_collection(
    collection_name="sugar_makeup_products",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)


#uploading vectors to data collection
print("[INFO] Creating vectors...")
records = []
for idx, sample in tqdm(enumerate(dataset['makeupProducts']), total=len(dataset['makeupProducts'])):
    if True:
        records.append(models.Record(
                id=idx, vector=encoder.encode(data_to_embd[idx]).tolist(), payload=sample
            ) ) 
    
print("[INFO] Uploading vector data to data collection...")   
qdrant.upload_records(
    collection_name="sugar_makeup_products",
    records=records,
)
print("[INFO] Successfully uploaded!")



# from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
# from qdrant_client import models, QdrantClient
# from tqdm import tqdm



# #instantiate qdrant client
# print("[INFO] Client created...")
# qdrant = QdrantClient("localhost", port=6333)


# #download dataset from hugging face
# print("[INFO] Loading dataset...")
# dataset = load_dataset("TrainingDataPro/asos-e-commerce-dataset")

# #process dataset
# train_ds = dataset['train']
# data = []
# req_columns = ['name','size','price','category','color','description']

# print("[INFO] Processing dataset...")
# for i in tqdm(train_ds, total=len(train_ds)):
#     data.append({col:i[col] for col in req_columns})

# #importing the GTE model
# print("[INFO] Loading encoder model...")
# encoder = SentenceTransformer('thenlper/gte-small')

# #creating data collection in qdrant
# print("[INFO] Creating a data collection...")
# qdrant.recreate_collection(
#     collection_name="e-shopping",
#     vectors_config=models.VectorParams(
#         size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
#         distance=models.Distance.COSINE,
#     ),
# )

# #uploading vectors to data collection
# print("[INFO] Uploading data to data collection...")
# records = []
# for idx, sample in tqdm(enumerate(data[:1000]), total=len(data[:1000])):
#     if sample['description']:
#         records.append(models.Record(
#                 id=idx, vector=encoder.encode(sample["description"]).tolist(), payload=sample
#             ) ) 
    
    
# qdrant.upload_records(
#     collection_name="e-shopping",
#     records=records,
# )
# print("[INFO] Successfully uploaded!")