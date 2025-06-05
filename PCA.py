import seaborn as sns , pandas as pd
from sklearn.datasets import load_iris; from sklearn.decomposition import PCA
pca = PCA(2)
data=pd.DataFrame(pca.fit_transform(load_iris().data),columns=['pc1','pc2'])
data['label']=load_iris().target
sns.scatterplot(x=data['pc1'], y=data['pc2'],hue=pd.Categorical.from_codes(data['label'],load_iris().target_names))
plt.title('PCA on iris dataset')
plt.show()
