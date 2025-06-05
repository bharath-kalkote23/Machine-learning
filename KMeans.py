import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x,y = load_breast_cancer(return_X_y=True)
x=StandardScaler().fit_transform(x)
model= KMeans(n_clusters=2)
pre=model.fit_predict(x)
print(classification_report(y,pre))
print(confusion_matrix(y,pre))
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=pre)
