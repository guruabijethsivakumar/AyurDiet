from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "food_vector_db_bge"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)

docs = list(db.docstore._dict.values())

print("Total docs:", len(docs))
print("\n=== SAMPLE METADATA (first 10 docs) ===\n")

for d in docs[:10]:
    print(d.metadata)
    print("-----------------------------------")
