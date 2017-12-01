from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

import extractedStats
from sklearn import neighbors, svm

vector_i = extractedStats.load_pokemon_images('TestImages')
vector_s = extractedStats.load_training_stats('TrainingMetaData.csv')
vector_y = extractedStats.load_training_labels('TrainingMetaData.csv')
vector_train = extractedStats.load_test_stats('UnlabeledTestMetaData.csv')
n_neighbors = 18

# Question 2 part E:
# Result of cross validation: [ 0.22058824  0.13846154  0.125       0.15        0.21666667  0.20338983
#  0.15517241  0.14035088  0.21428571  0.12962963]
# Average is 16.935%
#classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

# Question 2 part F:
# Result of cross validation: [ 0.16176471  0.15384615  0.203125    0.2         0.18333333  0.16949153
#  0.25862069  0.15789474  0.14285714  0.12962963]
# Average is 17.605%
#classifier = svm.SVC(kernel='linear')

# Question 2 part G:
# Result of cross validation: [ 0.13235294  0.12307692  0.15625     0.21666667  0.21666667  0.15254237
#  0.10344828  0.14035088  0.19642857  0.09259259]
# Average is 15.303%
#classifier = svm.SVC(kernel='poly', degree=3)

# Question 2 part H:
# Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.13333333  0.15254237
#  0.13793103  0.12280702  0.125       0.14814815]
# Average is 13.642%
#classifier = svm.SVC(kernel='rbf')

classifier = MLPClassifier(solver='sgd')
print('stepa')
print(vector_y)
print(vector_s)
cv_results = cross_validate(classifier, vector_s, vector_y, cv=10, return_train_score=False)
sorted(cv_results.keys())
print(cv_results['test_score'])
