from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("Hello Bangladesh", convert_to_numpy=True)
print(vector.shape)   # should print (384,)
