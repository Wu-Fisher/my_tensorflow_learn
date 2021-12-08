## 1.迭代器和fit中重要参数理解
    fit.steps_per_epoch=每个epoch中的step数
    generator.batch_size=每一个step中抽取的样本
    两者相乘应该为样本总数
    fit.epochs在数据增强下，可以理解为每一个epoch中，
    出来的样本数据总体都不一样，每次迭代这些不一样的总体样本的次数