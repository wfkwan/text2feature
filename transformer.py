from embedding import WordEmbedding
from sklearn.base import TransformerMixin

'''
source: https://www.kaggle.com/nhrade/text-classification-using-word-embeddings
'''

class Transformer(WordEmbedding, TransformerMixin):
	
	def __init__(self):
		pass

	def fit(self, X, y=None):
        return self

    def _doc_mean(self, doc):
        return np.mean(np.array([self._E[w.lower().strip()] for w in doc if w.lower().strip() in self._E]), axis=0)

    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def plot_roc(clf, X_test, y_test):
	    y_pred = clf.predict(X_test)
	    fpr, tpr, _ = roc_curve(y_test, y_pred)
	    plt.plot(fpr, tpr)
	    plt.xlabel('FPR')
	    plt.ylabel('TPR')
    
	def print_scores(clf, X_train, y_train, X_test, y_test):
	    clf.fit(X_train, y_train)
	    y_pred = clf.predict(X_test)
	    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred)))
	    print('AUC score: {:3f}'.format(roc_auc_score(y_test, y_pred)))

