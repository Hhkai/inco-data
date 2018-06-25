## 算法的简单介绍

1. 

> 训练集: 不完整的数据集; 测试集: 完整的数据集

> 训练流程: 将不完整的数据集分类, 在每个类别下他们被认为是完整的(因为我们只考虑了部分列而不是全部列). 然后得到很多个分类器. 

> 预测流程: 对于待分类的样本, 每个分类器都会为他作出一个分类结果, 然后对结果投票. 

2. 

> 训练集: 不完整的数据集; 测试集: 完整的数据集

> 训练流程: 将不完整的数据集分类, 选取其中的一部分类别来得到分类器. 

> 预测流程: 对于待分类的样本, 我们选中的每个分类器都会得到一个结果, 对结果投票. 

3. 

> 训练集: 不完整的数据集, 这个不完整的数据集的每个数据的完整版本; 测试集: 完整的数据集. 

> 训练流程: 将不完整的数据集分类, 选取其中的一部分得到分类器. 然后用他们的完整版本来做一次逻辑回归, 得到每个分类器的权重. 

> 预测流程: 对于待分类样本, 用我们选取的分类器来投票, 权值为前文提到的逻辑回归得到的权值. 

4. (来自文章Classification for Incomplete Data Using Classifier Ensembles)

> 训练集: 不完整的数据集; 测试集: 完整数据集

> 训练流程: 将不完整的数据集分类, 每一类得到一个分类器. 其中, 每个分类器都由Adaboost+ANN得到. 

> 预测流程: 对于待分类样本, 用所有分类器投票. 

5. (来自文章Multi-granulation Ensemble Classification for Incomplete Data)

> 训练集: 不完整的数据集; 测试集: 不完整的

> 训练流程: 将不完整的数据集分类(文中提到了树形结构, 但后面又加了合并过程, 这个合并过程使得这个树形结构事实上无效), 每一类得到一个分类器. 

> 预测流程: 对于一个待分类样本, 得到它的缺失类别A. 首先用这个缺失类别的分类器预测, 然后给A添加更多的缺失项, 并用对应的分类器进行预测, 最后投票.

6. 

> 训练集: 不完整的数据集; 测试集: 不完整的

> 训练流程: 将不完整的数据集分类, 每一类得到一个分类器. 其中, 每个分类器都用Adaboost的方法整合了其*下属*的分类器, 这里下属的意思是在其缺失类型上缺得更多. 

> 预测流程: 对于一个待分类的样本, 用其对应的分类器作预测. 

7. 

> 训练集: 不完整的数据集; 测试集: 不完整的

> 训练流程: 将不完整的数据集分类, 每一类得到一个分类器. 其中, 每个分类器都用Adaboost的方法整合了多个分类器, 那么分类器是比它缺失类型更少的. 

> 预测流程: 对于一个待分类的样本, 用其对应的分类器预测. 

## 为什么会出现-1

观察6和7的预测过程, 我们是用其对应分类器作预测的, 也就是说, 如果我们要预测一个缺(1, 3)的不完整数据, 那么必定要有对应的分类器. 由训练流程知, 我们有一个(1, 3)的分类器当且仅当训练集出现过(1, 3)的缺失样本. 

那么, 当我们将此算法应用于完整的测试集上时, 如果训练集中没有"不缺"的缺失属性的样本, 就没有对应完整样本的分类器, 所以无法预测. 

## 为什么别的算法不会有-1

换句话说, 别的算法不会出现无法预测的情况. 因为总的来说, 最后这两个算法, 根本上讲, 只用了一个分类器来预测(虽然这个分类器整合了多个分类器), 如果待预测样本所需要的分类器不存在, 就会无法预测. 而其他算法, 因为是用多个分类器投票得到结果, 所以不会出现"分类器不存在"的情形. 