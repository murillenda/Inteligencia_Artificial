import pandas as pd
import matplotlib.pyplot as plt
import pydotplus as pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image

nameColumns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
dataFrame = pd.read_csv('iris.csv', names=nameColumns)

# Visualizar primeiras linhas da tabela
#a = dataFrame.head()

# Quantas flores tem cada categoria
b = dataFrame['Class'].value_counts()

# Teste Fumaça
print('Linhas: %d, Colunas: %d' % (len(dataFrame), len(dataFrame.columns)))

# Criação de Features
dataFrame['SepalArea'] = dataFrame['SepalLength'] * dataFrame['SepalWidth']
dataFrame['PetalArea'] = dataFrame['PetalLength'] * dataFrame['PetalWidth']

'''
# Analise de dados estatísticos do dataframe
print(dataFrame['SepalArea'].mean())
print(dataFrame['SepalArea'].min())
print(dataFrame['SepalArea'].max())
print(dataFrame['SepalArea'].mode())
print(dataFrame['SepalArea'].median())
print(dataFrame['SepalArea'].mean(), 2)  # arrumar numeros flutuantes
'''
# Criação de outras features
dataFrame['SepalLengthAboveMean'] = dataFrame['SepalLength'] > dataFrame['SepalLength'].mean()
dataFrame['SepalWidthAboveMean'] = dataFrame['SepalWidth'] > dataFrame['SepalWidth'].mean()
dataFrame['PetalLengthAboveMean'] = dataFrame['PetalLength'] > dataFrame['PetalLength'].mean()
dataFrame['PetalWidthAboveMean'] = dataFrame['PetalWidth'] > dataFrame['PetalWidth'].mean()

# Recuperar o nome das colunas que formam as variáveis independentes
features = dataFrame.columns.difference(['Class'])
# features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'SepalArea', 'PetalArea', 'SepalLengthAboveMean',
            # 'SepalWidthAboveMean', 'PetalLengthAboveMean', 'PetalWidthAboveMean']
print('Features: ', features)
print(dataFrame.head())
# Gerar gráfico de quantas flores cada um tem
c = b.plot.pie(title='Gráfico de Quantidade', fontsize=20, table=True, figsize=(10, 6), autopct='%1.1f%%', startangle=90)
plt.show()

# Criar armazenadores para as variáveis independentes (X) e para a dependente (y)
X = dataFrame[features].values
y = dataFrame['Class'].values

# Criar variáveis com valores que não estão entre os do treinamento para averiguar se a árvore foi criada corretamente
# Iris-setosa
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]
# Iris-versicolor
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]
# Iris-virginica
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]

# Criação da árvore de decisõa
# classifier_dt = DecisionTreeClassifier(max_depth=3)
classifier_dt = DecisionTreeClassifier(random_state=10, criterion='gini',max_depth=3)
classifier_dt.fit(X, y)

# Testar se a árvore de decisão consegue classificar corretamente os exemplos
print(classifier_dt.predict([sample1, sample2, sample3]))
'''
# Exibição da árvore de decisão criada
dot_data = tree.export_graphviz(classifier_dt, out_file=None, feature_names=features, class_names=dataFrame.Class.unique())

#escrever a imagem
graph = pydotplus.graph_from_dot_data(dot_data)

#mostrar imagem
Image(graph.create_png())'''