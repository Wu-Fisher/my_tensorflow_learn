# DAY6 第四章
## 4.1 四个分支
### 监督学习
    标注样本
    二分类问题，多分类问题，标量回归问题
    序列生成，语法树预测，目标检测，图像分割
### 无监督学习
    没有目标的情况
    降维，聚类

### 自监督学习
    没有人工标注的标签的监督学习，通过启发式算法生成
    自编码器
## 4.2评估
    训练集，验证集，测试集
    简单留出，K折，打乱数据重复K折
    数据代表性：
        如果你的留出分类使得训练集和测试集中有明显的不相交，（例如训练集只有0-7分类，测试集8-9分类）应该先打乱
    时间箭头：
        对于天气问题，测试集时间应该晚于训练集
    数据冗余：   
        保证训练集和验证集没有交集
## 4.3
### 一.数据预处理
        向量化：记得是浮点数，one-hot编码
        标准化：将输入数据满足 取值较小，同质性
                类似标准化
        处理缺失值：0意味着缺失数据
### 二.特征工程
        提取优化特征，避免资源浪费
## 4.4过拟合与欠拟合
   优化，泛化
### 正则化
####   1.减小模型大小（隐藏单元）
        一般从小到大，再看过拟合时间和损失变化速度
        一般只能选择到一定范围
####   2.添加权重正则化
        强制让模型权重取较小的值，权重值分布更规律,方法上向网络损失函数添加与较大权重值相关的成本
        
        L1正则化和L2正则化
        L2正则化例子

        损失会提高训练损失，但是会使验证损失更加稳定，不容易过拟合
```python
model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))
```

####    3.添加dropout正则化
        在训练过程中，对某一层使用dropout，将该层的一些输出特征舍去，dropout rate是舍去比例（0.2～0.5）
        测试时，由于没有被舍弃，所以输出值要按照dropout比例缩小，或者把测试输入值/（dropout rate）
```python
    model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
```
## 常见总结
    (问题，最后一层，loss_func)
    二分类问题  sigmoid   binary_crossentropy
    多分类，单标签   softmax categorical_crossentropy
    多分类多标签 sigmoid binary_crossentropy
    回归任意值  无  mse
    回归0～1   sigmoid   mse or binary_crossentropy
        

    
        