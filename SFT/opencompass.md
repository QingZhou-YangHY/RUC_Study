# OpenCompass

1.✨基础环境安装,已clone  
推理后端,API似乎不用安  

2.✨额外的数据集也没有  
我们这次要用的是自建以及第三方数据集,已经手动下载解压.在/home/chenzhipeng/yhy/opencompass/data路径下.里面是.csv和json文件(json只有一个,剩下的都是.csv)  
但是在/home/chenzhipeng/yhy/opencompass/opencompass/datasets路径下又找到ceval.py和mmlu.py  
### TODO1  👈这里希望可以讲解一下

3.✨配置  
选择要评估的模型和数据集  
- 命令行: 对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。  
通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。(X)
- 命令行:用户可以使用 --models 和 --datasets 结合想测试的模型和数据集。模型和数据集的配置文件预存于 configs/models 和 configs/datasets 中。用户可以使用 tools/list_configs.py 查看或过滤当前可用的模型和数据集配置。(X)
- 配置文件:配置文件是以 Python 格式组织的，并且必须包括 datasets 和 models 字段.我们只需将配置文件的路径传递给 run.py：
python run.py configs/eval_chat_demo.py --debug  
数据集配置通常有两种类型：ppl 和 gen，分别指示使用的评估方法。其中 ppl 表示辨别性评估，gen 表示生成性评估。对话模型仅使用 gen 生成式评估。

### TODO2:如何去写配置文件? 
- 通过 继承机制 引入所需的数据集和模型配置，并以所需格式组合 datasets 和 models 字段。  
    - 很多.csv文件和json文件怎么像demo一样~~import进去?~~ 👈这里不太明白
- 通过定义变量的形式指定每个配置项 👉[学习配置文件](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/config.html#id3)
- 继承配置文件时使用 read_base 上下文管理器
- 当读取配置文件时，使用 MMEngine 中的 Config.fromfile 进行解析
- 数据集配置文件可以跳过,直接使用mmlu和c-eval完成复现😘
- 数据集配置文件似乎都是.py?(文档上是这么写的,我这里都是.csv json)也许我现在这个叫数据集，还不是配置(大概率是对的),path这么多怎么写?不确定这是不是一个好问题👉[配置数据集](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/datasets.html)
- 不是HuggingFace，不是API，是自定义模型 三种汇总👉[准备模型](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/models.html) 自定义模型点这里👉[支持新模型](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/new_model.html)  
- 似乎是正确的，所以后面是否要在方法里面重新写?不是很懂
- python配置/脚本之前没有搞明白，希望能深入了解并掌握💪

4.✨评估  
让模型从数据集产生输出并衡量这些输出与标准答案的匹配程度  
输出将被重定向到输出目录 outputs/demo/{TIMESTAMP}  
任何后端任务失败都只会在终端触发警告消息  
run.py中的更多参数看官方文档  
.py 配置文件作为任务相关参数，里面需要包含 datasets 和 models 字段  
### TODO3:似乎只差这一个.py配置文件就完成任务了 😊


5.✨可视化  
将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件  
要有summarizer,用于按照自己客制化的期望来输出，summarizer 出现在 config 中
- dataset_abbrs展示列表项，要不然全都输出了  
- summary_groups汇总指标配置
    - name: (str) 汇总指标的名称
    - ...看官方文档
- 注意，我们在 configs/summarizers/groups 路径下存放了 MMLU, C-Eval 等数据集的评测结果汇总，建议优先考虑使用(确实有,看到了)  
打印评测结果的过程可被进一步定制化，用于输出一些数据集的平均分 (例如 MMLU, C-Eval 等)👉[YuLan-Chat](https://github.com/RUC-GSAI/YuLan-Chat)