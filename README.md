
# [数据集](#目录)


 ```text
└─data
    ├─train                 # 训练数据集
       ├0
        ├36011.jpg
        ├...
       ├1
       ├2
       ├3
       ├4
       ├5
       ├6
       ├7
       ├8
       ├9
    └─test                   # 评估数据集
       ├img_0.jpg
       ├img_1.jpg
       ├...

 ```


# 脚本及样例代码

```text
├── food_conformer
  ├── README_CN.md                        // 相关说明
  ├── src
      ├──configs                          // RepVGG的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ├──repvgg.py                    // repvgg模型文件
          ┕──layers                       // 相关工具层定义文件
          ┕──RepVGG                          // RepVGG定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
  │   ├──tools                            // 工具文件夹
          ├──callback.py                  // 自定义回调函数，训练结束测试
          ├──cell.py                      // 一些关于cell的通用工具函数
          ├──criterion.py                 // 关于损失函数的工具函数
          ├──get_misc.py                  // 一些其他的工具函数
          ├──optimizer.py                 // 关于优化器和参数的函数
          ┕──schedulers.py                // 学习率衰减的工具函数
  ├── train.py                            // 训练文件
  ├── eval.py                             // 评估文件
  ├── preprocess.py                       // 推理数据集与处理文件
  ├── postprocess.py                      // 推理精度处理文件
  ├── train_log.log                       // 训练日志
  ├── result_sort.txt                     // 推理结果文件
```

# 推理过程

## 推理脚本



```bash
python eval.py --run_openi True --data_url /home/ma-user/work/data --batch_size 1  --num_classes 10 --pretrained True --ckpt_url ./ckpt_0/best.ckpt --eval 1

```
