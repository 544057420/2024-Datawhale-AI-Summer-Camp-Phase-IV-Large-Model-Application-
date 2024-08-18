# 2024-Datawhale-AI-Summer-Camp-Phase-IV-Large-Model-Application
2024Datawhale AI夏令营第四期方向2：大模型开发与应用
![demo1.0](https://github.com/544057420/2024-Datawhale-AI-Summer-Camp-Phase-IV-Large-Model-Application-/blob/main/demo12024-08-18%20104815.jpg)
基于浪潮“源”大模型的AI心理咨询师与危机干预助理

## 课程链接：<https://linklearner.com/activity/14/11/25>

## 模型下载地址: www.modelscope.cn/datasets/Datawhale/AICamp_yuan_baseline

## 数据集：

[97.5K rows]<https://hf-mirror.com/datasets/chansung/mental_health_counseling_conversations_merged>

[3.5k rows]<https://hf-mirror.com/datasets/Amod/mental_health_counseling_conversations>

[中文心理健康支持对话 · 数据集(SmileChat)与大模型(MeChat)]<https://github.com/qiuhuachuan/smile>

### 精神医学与心理医学资料(由于版权原因，部分专著在此无法公开其Z-lib获取链接)
[危机干预热线与自杀干预知识](https://mp.weixin.qq.com/s/SCj8hyeZxJGgMGwYTyzVmQ)

[疾病与状况 - 妙佑医疗国际](mayoclinic.org)

[MSD诊疗手册专业版](https://www.msdmanuals.cn/professional)

[PubMed](https://pubmed.ncbi.nlm.nih.gov/?Db=pubmed)

Cognitive-Behavioral Strategies in Crisis Intervention, Third Edition (Frank M. Dattilio, Arthur Freeman) (Z-Library)

Psychological Interventions in Times of Crisis (Laura Barbanel EdD  ABPP etc.) (Z-Library)

伯恩斯新情绪疗法 (大卫.伯恩斯) (Z-Library)

大学生心理危机干预指南 (金晓明) (Z-Library)

危机和创伤中成长：10位心理专家危机干预之道（如何面对新冠疫情等重大创伤？12位国内知名心理学专家教你如何与危机相处） (方新  主编 [方新... (Z-Library)

自闭症问题行为干预 (自闭症前沿研究丛书) (园山繁树  裴虹) (Z-Library)

### 需要危机干预识别的标志

https://github.com/Chenxiaosen-Neo/CompleteSuicideManual2022

https://github.com/YuriMiller/CompleteSuicideManual-Zh_CN

### 心理测评类网站
1. **PsyToolkit**
   - 提供各种心理学测试和认知功能测试。
   - 网址：[PsyToolkit](http://psytoolkit.org/)

2. **Mental Health America (MHA) Screening**
   - 提供多种心理健康筛查工具。
   - 网址：[MHA Screening](https://www.mhanational.org/)

3. **Patient.info**
   - 提供一些基本的健康和心理状态自我评估工具。
   - 网址：[Patient.info](https://www.patient.info/)

4. **7 Cups**
   - 提供情绪健康测试和其他心理健康相关的资源。
   - 网址：[7 Cups](https://www.7cups.com/)

5. **Your Mental Health**
   - 提供一些基本的心理健康测试。
   - 网址：[Your Mental Health](https://www.yourmentalhealth.org/)

6. **Self-Check Tools**
   - 提供自我检查工具，包括焦虑和抑郁的自我评估。
   - 网址：[Self-Check Tools](https://www.selfchecktools.com/)

7. **Mind Diagnostic Test**
   - 提供一些心理健康状态的自我评估。
   - 网址：[Mind Diagnostic Test](https://www.mind.org.uk/information-support/mental-health-problems/self-help-and-everyday-life/self-check-mental-health-diagnostics/)

8. **The Big White Wall**
   - 提供在线社区支持和心理健康测试。
   - 网址：[The Big White Wall](https://www.thebigwhitewall.com/)

9. **Screening Tests**
   - 提供多种心理健康筛查测试。
   - 网址：[Screening Tests](https://www.screeningtests.com/)

10. **National Institute of Mental Health (NIMH)**
    - NIMH 提供一些心理健康信息和资源，虽然不直接提供在线测试，但有链接到其他筛查工具。
    - 网址：[NIMH](https://www.nimh.nih.gov/)
11.  **https://www.psyctest.cn/**
    - 赛可心理测试官方网站
    - 网址：https://www.psyctest.cn/
## 大模型相关资料：

大模型基础 so-large-lm <https://github.com/datawhalechina/so-large-lm>

Datawhale开源教程，掌握大模型理论知识！

开源大模型食用指南 self-llm<https://github.com/datawhalechina/self-llm>

Datawhale开源教程，速通大模型部署&微调！

大模型白盒子构建指南  tiny-universe<https://github.com/datawhalechina/tiny-universe>

Datawhale开源教程，围绕大模型全链路的“手搓”大模型指南！

动手学大模型应用开发 llm-universe<https://github.com/datawhalechina/llm-universe>

Datawhale开源教程，一个面向小白开发者的大模型应用开发教程！

其他模型下载：hugging face 镜像<https://hf-mirror.com/>

## 安装方法
应答模型基于`Yuan2-2B`，向量模型基于`bge-small-zh-v1.5`,GUI界面需要安装 `streamlit`
### 环境准备（含模型下载）
方法1

安装依赖包

```python
pip install -r requirements.txt
```

下载模型
```python
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='.')
```

方法2
```python
# 查看已安装依赖
! pip list
# langchain安装
!pip install pypdf faiss-gpu langchain langchain_community langchain_huggingface streamlit==1.24.0
# 安装 streamlit
! pip install streamlit==1.24.0
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='.')
# 导入所需的库
from typing import List
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
```
AfterEMOdemo.py代码已经完成了加载嵌入模型和计算嵌入的任务，因此不需要再定义一个EmbeddingModel类。

### 进入GUI
```python
streamlit run AfterEMOdemo.py --server.address 127.0.0.1 --server.port 6006
```

