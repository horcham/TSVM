## TSVM
TSVM算法实现，参照《机器学习》（周志华） 13.3节

## Usage
 - 构建TSVM
   ```
   model = TSVM()
   ```
 - TSVM的初始化
   ```
   model.initial(kernel = 'linear')
   ```
   `kernel`为所使用的svm的核， 默认为`linear`

   若要加载已有模型，则
   ```
   model.load(model_path)
   ```
   `model_path`为TSVM所存放的路径
 - 训练TSVM
   ```
   model.train(X1, Y1, X2)
   ```
   其中，`X1`为有标签数据，其标签为`Y1`，`X2`为无标签数据。 `X1`，`X2`为`numpy.array`，shape为`[n,m]`，
   `Y1`为`numpy.array`，shape为`[n, ]`，其中，`n`代表样本个数，`m`代表属性个数
 - 使用TSVM预测
   ```
   Y_hat = model.predict(X)
   ```
   其中，`Y_hat`为`numpy.array`，shape为`[n, ]`
 - 计算TSVM准确率
   ```
   accuracy = model.score(X, Y)
   ```
 - 保存模型
   ```
   model.save(model_path)
   ```
