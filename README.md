## 依赖项

- transformers
- pandas
- torch
- numpy

## 数据集

+ 需要准备一个包含文本数据的CSV文件。在代码示例中，我们提供了两个示例CSV文件：`Mashups.csv`和`APIs.csv`，读者可以根据需要选择其中一个文件，或使用自己的CSV文件。

## 输出文件

+ BERT编码后的数据将以`.npy`文件的形式保存在指定的输出文件中。

## 注意事项

+ 在代码中，根据数据和需求，需要更改以下部分：

  - 数据文件的路径：修改`data = pd.read_csv("datasets/APIs.csv", encoding='iso-8859-1')`中的文件路径。

  - 输出文件的名称：在`np.save("data_bert_encoding_APIs", bianma_APIs, allow_pickle=True)`中指定要保存BERT编码的输出文件名称。

+ 请确保数据文件是适当格式的CSV文件，并且包含一个名为`'Description'`的列，以便代码能够正确读取文本数据。


#### Contact
Email: 2635814455@qq.com (陈婉君)
