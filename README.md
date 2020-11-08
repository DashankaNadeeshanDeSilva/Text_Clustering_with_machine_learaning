# Text_Clustering_with_machine_learaning

Movie data including titles, synopses and genres are clustered using 3 machine learning algorithms. It has used both IMDB and WIKI syposes of movies for this classification along with other information.

**Clustering is performed with:**

1. K-meand algorithm
2. Hierarchical clustering on the corpus (using Ward clustering)
3. Topic modeling using Latent Dirichlet Allocation (LDA)

Follwing steps are involved in the script:

1. Data preparation
2. Tokenizing and stemmung synopses - Splits the synopsis into a list of its respective words or tokenize corpus
3. TFIDF and document similarity - Define term frequency-inverse document frequency (tf-idf) vectorizer object (parameters) the convert the synopses list 
into a tf-idf matrix using the vectorizer object and use cosine distance between each text as a measure of similarity
4. 
