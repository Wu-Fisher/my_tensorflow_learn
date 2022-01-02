# Summary

密集连接层和卷积层的区别：卷积层学到的是局部模式
1.平移不变性
2.模式的空间层次结构

对于Conv2D层，使用padding参数表示是否填充

卷积步幅，
最大池化运算 2*2 步幅 2
卷积 3*3 布幅 1

采用下采样的原因
1.减少处理特征图的元素个数
2.通过让连续卷积层的观察窗口越来越大（窗口覆盖原始输入的比例越来越大）引入空间过滤器的层级结构

对于更大的文件，特征图的深度在逐渐增大但特征图的尺寸在逐渐减小

数据增强：从现有训练样本中生成更多的训练数据，不会查看两次完全相同的图像，让模型能观察到数据的更多内容，从而有更好的泛化能力
*利用imagedatagenerator来设置数据增强
*加入dropout层