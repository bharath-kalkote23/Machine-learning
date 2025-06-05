import matplotlib.pyplot as plt,numpy as np
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
Data,tra=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,t_test=train_test_split(Data,tra,test_size=0.2)
clsf=DecisionTreeClassifier().fit(x_train,y_train)
print(accuracy_score(t_test,clsf.predict(x_test)))
val=(clsf.predict(np.array([x_test[0]]).reshape(1,-1)))
print("benign" if val==1 else "malignant")
tree.plot_tree(clsf,filled=True)
plt.show()
