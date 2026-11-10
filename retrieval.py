def retrieve(query, db, k=8):
    return db.similarity_search(query, k=k)