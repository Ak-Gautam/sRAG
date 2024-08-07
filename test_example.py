import numpy as np
from collections import Counter


corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new about history, art, or science.",
    "Attend a live music concert and feel the rhythm of your favorite genre, be it rock, jazz, or classical.",
    "Go for a hike in nature and admire the beauty of the natural scenery, from mountains and forests to deserts and beaches.",
    "Have a picnic with friends and share some laughs, good food, and great company in the outdoors.",
    "Explore a new cuisine by dining at an ethnic restaurant and tantalize your taste buds with exotic flavors.",
    "Take a yoga class and stretch your body and mind, promoting relaxation and inner peace.",
    "Join a local sports league and enjoy some friendly competition, while getting exercise and socializing with others who share your passion for the sport.",
    "Attend a workshop or lecture on a topic you're interested in, to expand your knowledge and gain new skills.",
    "Visit an amusement park and experience the thrill of riding roller coasters, bumper cars, and other exciting attractions.",
    "Go stargazing on a clear night and marvel at the wonders of the universe.",
    "Volunteer at a local charity and give back to your community.",
    "Learn a new language and open yourself up to new cultures and experiences.",
    "Take a road trip and explore new places, experiencing the beauty and diversity of the world around you.",
    "Go camping under the stars and reconnect with nature.",
    "Read a book and get lost in a captivating story.",
    "Binge-watch a tv show or movie series and enjoy a relaxing escape.",
    "Try a new hobby or activity, like painting, pottery, or playing a musical instrument.",
    "Spend time with loved ones and create lasting memories.",
    "Take a relaxing bath and unwind after a long day."
]

def cosine_similarity(query, document):
    # Tokenize and count words
    query_counts = Counter(query.lower().split())
    doc_counts = Counter(document.lower().split())
    
    # Get the union of words
    all_words = set(query_counts.keys()) | set(doc_counts.keys())
    
    # Create vectors
    query_vec = np.array([query_counts.get(word, 0) for word in all_words])
    doc_vec = np.array([doc_counts.get(word, 0) for word in all_words])
    
    # Compute dot product and norms
    dot_product = np.dot(query_vec, doc_vec)
    norm_query = np.linalg.norm(query_vec)
    norm_document = np.linalg.norm(doc_vec)
    
    # Compute and return the cosine similarity
    if norm_query == 0 or norm_document == 0:
        return 0.0  # Handle edge case where one vector is zero
    return dot_product / (norm_query * norm_document)

def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        similarity = cosine_similarity(user_input, doc)
        similarities.append(similarity)
    return corpus_of_documents[similarities.index(max(similarities))]

user_input = "I like to hike"
return_response(user_input, corpus_of_documents)