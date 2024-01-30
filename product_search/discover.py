from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient


#instantiate qdrant client
print("[INFO] Client created...")
qdrant = QdrantClient("localhost", port=6333)

#importing the GTE model
print("[INFO] Loading encoder model...")
encoder = SentenceTransformer('thenlper/gte-small')

pref = "I wish to buy a wool coat that is very cozy"

like = ["Fashion Wear","Woolen Cloths", "comfortable"]
dislike = ["Batteries", "silks", "toys"]

context = [models.ContextExamplePair(positive=encoder.encode(l).tolist(), negative=encoder.encode(d).tolist()) for (l,d) in list(zip(like, dislike))]

discover_queries = [
    models.DiscoverRequest(
        target=encoder.encode(pref).tolist(),
        context=context,
        limit=2,
    ),
]

print(discover_queries)