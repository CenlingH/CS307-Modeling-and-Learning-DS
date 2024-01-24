# M0

### Machine Learning tasks

1. 分类：
   1. supervised learning: 有东西可以学习
      1. classification
      2. regression
   2. unsupervised learning
      1. density estimation
      2. clustering
      3. novelty and outlier detection
      4. dimension reduction
   3. reinforcement learning

### Simple Methods and Metrics

1. Baseline Methods:

   ```python
   DummyClassifier # most common category
   DummyRegressor # use median, arbituary constant(都跟train数据无关的)
   ```

2. Metrics(regression metrics)

   - accuracy:

     $Accuracy(y,\hat y)=\frac 1 n \sum_{i=1}^{n}I(y_i=\hat y_i)$

     I 相等为1，不等为0

     ```python
     def accuracy(y_true,y_pred):
     	return np.mean(y_true==y_pred)
     def misclassification(y_true,y_pred):
         return np.mean(y_true!=y_pred)  # 不是miss，或者求np.sum之后除掉len(y_true)
     ```

   - root mean square error:

     $RMSE(y,\hat y)=\sqrt{\frac1n\sum^{n}_{i=1}(y_i-\hat y_i)^2}$

   - mean absolute error:

     $MAE(y,\hat y)=\frac1n\sum_{i=1}^n|y_i-\hat y_i|$

   - mean absolute percent error:

     $MAPE(y,\hat y)=\frac1n\sum_{i=1}^n\frac{|y_i-\hat y_i|}{|y_i|}$

   - coefficient of determination:

     $R^2(y,\hat y)=1-\frac{\sum_{i=1}^n(y_i-\hat y_i)^2}{\sum_{i=1}^n(y_i-\bar y)^2}$

   - max error:

     $Max Error=max(|y_i-\hat y_i|)$