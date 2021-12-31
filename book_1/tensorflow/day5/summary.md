# 小结
## ta    
    1.回归问题损失函数用均方误差（mse）
    2.评估指标也不能用精度，二要用平均绝对误差MAE
    3.特征取值范围不同时，预处理（本次使用的标准化）
    4.数据少，使用K折验证，隐藏层也较少

## tips
    1. data.mean 均值 data.std 标准差
    2. concatenate()拼接数组
    3. // 整除，向下取整
    4. model.fit(verbose = 0) <=静默模式，不输出epoches信息
    5.Backend TkAgg is interactive backend. Turning interactive mode on.
       1.消除提示的方法如下：在pycharm的setting中找到Version Control _> subversion，将Enable interactive mode选项选中，后续就不会再出现
        2.plt.interactive(True) 
    6. no show() !!!  plt.savefig("name.png",dpi=200)