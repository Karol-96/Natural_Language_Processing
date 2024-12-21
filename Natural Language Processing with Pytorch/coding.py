




from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns


tf_idf_vectorizer = TfidfVectorizer()
tf_idf= tf_idf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tf_idf,annot=True,cbar=False,xticklabels=vocab, yticklabels=['Sentence 1','Sentence 2'])
