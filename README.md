# SwinDNet
step1:数据准备
将训练的image放到data/JPEGImages文件夹中，label放到data/SegmentationClass中
data/test文件夹中放测试image和label
data/val文件夹中放验证image和label

step2：训练  
运行train.py进行训练,训练得到的模型参数保存在params文件夹下

step3：测试
运行test.py进行预测，预测得到的结果保存在result文件夹下

step4:评估
FLOPs.py文件负责计算模型的浮点数运算量和计算量
inferenceTimeCaculate文件负责计算推理时间
evalution/get_evalution.py负责计算IoU和Dice
