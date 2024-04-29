这个代码是我2023年暑假在NTU参加人工智能交流项目时进行医疗图像识别诊断时写的，所有代码均是我一人编写。
这个代码是对于肾脏CT图像进行识别诊断的深度学习方法实现，主要跑了当时几个比较火的model，如：ViT,Resnet,Googlenet等。
实验结果如下：
Model		      Epoch		  Train accuracy		Test accuracy
Resnet50 		   100			    84.78%			     46.91%
ViT			       100			    41.36%			     46.03%
Googlenet	     188			    85.33%			     90.18%

#代码介绍
整个代码分为data,model,train和test部分,其中：
dataset：编写好的dataset文件。
model：含有几个测试的模型pytorch框架实现。
train：编写了各个模型的训练代码，并且在每20epoch处进行模型参数保存。
test：编写了各个模型的测试代码。
Weights：为存储了各个模型的在各个epoch处保存的模型参数，方便载入。（需自行创建文件夹）

#数据集下载：CT of Kideney
kaggle：https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data

#运行
需要下载好数据集，同时修改相应载入数据集的路径以及Weights的model参数载入和保存路径即可，在train中选择相应的训练函数运行即可。

这是我学习接触人工智能半年时参加接触的项目代码，内部有很多代码结构不如GitHub上顶会论文中的简洁且有体系，但是这个代码让我更实在的接触了深度学习的编程，提高了我的编程能力。希望老师多多指教~
