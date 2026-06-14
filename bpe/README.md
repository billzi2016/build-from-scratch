# bpe

## 项目简介

`bpe` 是一个面向初学者的教学型项目，用来从头解释和实现 `Byte Pair Encoding` 的核心流程。

这里的重点不是直接调用现成 tokenizer，而是让读者真正看懂：

1. BPE 为什么要做子词切分
2. 初始最小单元是什么
3. 相邻 pair 频率是怎么统计的
4. merge 规则是怎么一步一步学出来的
5. encode 和 decode 是怎么工作的

---

## 这个项目里的 from scratch 是什么意思

`bpe` 和 `imagenet-ai` 不一样。

在这个项目里，`from scratch` 的意思是：

1. BPE 核心算法自己写
2. pair 统计自己写
3. merge 学习自己写
4. 词表构建自己写
5. encode / decode 自己写

也就是说，这里不能直接调用现成 BPE 库来替代核心过程。

---

## 可以使用什么

这个项目默认只使用基础能力，例如：

1. Python 标准库
2. `re`
3. `collections`
4. `argparse`
5. `pathlib`

这些能力只负责基础工程组织和文本清洗，不替代 BPE 的核心逻辑。

---

## 不可以使用什么

默认不允许：

1. `tokenizers`
2. `sentencepiece`
3. `tiktoken`
4. 任何现成 BPE 训练器
5. 任何直接导入后就能得到 merge / vocab 的现成实现

这个项目的重点是把 merge 过程教明白，而不是把第三方库包一层。

---

## 当前目录结构

```text
bpe/
├── README.md
├── bpe_prd.md
└── build_wheel/
    ├── main.py
    ├── cleaner.py
    ├── trainer.py
    ├── encoder.py
    ├── decoder.py
    ├── vocab.py
    └── utils.py
```

各部分职责如下：

1. `README.md`：快速说明这个项目在做什么
2. `bpe_prd.md`：约束项目边界和实现规则
3. `main.py`：串起整个教学流程
4. `cleaner.py`：负责基础文本清洗
5. `trainer.py`：负责 BPE 训练主流程
6. `encoder.py`：负责编码
7. `decoder.py`：负责解码
8. `vocab.py`：负责初始符号表示和词表操作
9. `utils.py`：负责展示和辅助函数

---

## 运行后能看到什么

运行 `build_wheel/main.py` 后，默认会按教学顺序打印：

1. 原始文本
2. 清洗后的文本
3. 词频统计
4. 初始字符级表示
5. 每轮 merge 的最佳 pair
6. 每轮 merge 形成的新 token
7. 最终 merges
8. 最终词表
9. 示例文本的 encode 结果
10. 示例 token 序列的 decode 结果

这意味着它不是只给最终答案，而是把整个过程展示出来。

---

## 如何运行

最基本的运行方式：

```bash
python bpe/build_wheel/main.py
```

也可以自己指定 merge 轮数：

```bash
python bpe/build_wheel/main.py --merges 8
```

也可以传入自己的训练文本和演示文本：

```bash
python bpe/build_wheel/main.py --merges 12 --text "low lower lowest new newer newest" --example "lowest newer"
```

---

## 教学边界

当前这个实现是教学版，不追求工业级 tokenizer 的全部复杂边界。

它默认聚焦：

1. 字符级初始表示
2. pair 统计
3. merge 学习
4. 最小可讲明白的 encode / decode

它暂时不优先解决：

1. 超大规模语料优化
2. 工业级性能问题
3. 复杂 Unicode 边界
4. 生产环境 tokenizer 兼容格式

---

## 适合谁看

这个项目适合：

1. 刚开始学 tokenizer 的人
2. 听过 BPE 名字，但没真正理解过 merge 过程的人
3. 想先把核心逻辑看懂，再去看复杂工业实现的人
