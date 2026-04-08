# AI Memory Store - 记忆存储系统

> 让 AI 拥有"认知沉淀"能力，不是分类柜，是关系网。

[English](./README.md) | 中文

---

## 🌟 理念

**记忆的本质不是存储，而是编织关联。**

两件看似无关的事，如果底层规律相同，就会在使用中自然产生关联。这套系统的核心不是"给记忆打标签分类"，而是"记录使用情境，让跨域规律自然沉淀"。

```
新对话 → 记录情境 → 被多次调用 → 跨域沉淀
```

---

## 📦 核心结构

### 两层记忆架构

| 层级 | 描述 | 特点 |
|------|------|------|
| 📝 Raw (原始记忆) | 初始记录 | 量大、冗余、允许碎片 |
| 🌊 Sediment (认知沉淀) | 跨域验证过的规律 | 精炼、内化、成为思维透镜 |

### 核心字段

```python
{
    "content": "记忆内容",
    "layer": "raw",           # 层级: raw / sediment
    "legacy": False,           # 是否为认知材料
    "usage_contexts": [],      # 使用情境记录
    "domain_crossings": 0,      # 跨域计数
    "emotion": "neutral",      # 情感标签
    "importance": 5,           # 重要性
    "type": "general"          # 类型: conversation/insight/milestone/daily/emotion/todo/general
}
```

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    双写机制                               │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐         ┌─────────────────────────┐ │
│  │   Markdown      │         │   TF-IDF + FAISS        │ │
│  │   (人类可读)     │ ←双写→  │   (语义搜索)             │ │
│  │   按日期存储     │         │   向量检索 + 过滤       │ │
│  └─────────────────┘         └─────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    沉淀机制                               │
├─────────────────────────────────────────────────────────┤
│  搜索记忆 → 记录当前情境 → 不同情境累计 → 3次跨域 → 自动沉淀│
└─────────────────────────────────────────────────────────┘
```

### 技术选型

| 组件 | 方案 | 理由 |
|------|------|------|
| 向量化 | TF-IDF | 无需外部模型，轻量高效 |
| 索引 | FAISS | 高效向量搜索 |
| 存储 | JSON | 简单可靠 |
| 双写 | Markdown + 向量库 | 人类可读 + 机器可搜 |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/ai-memory-store.git
cd ai-memory-store
```

### 2. 安装依赖

```bash
pip install scikit-learn faiss-cpu
```

### 3. 初始化

```bash
python3 -c "
from dual_memory import DualMemoryStore

store = DualMemoryStore(
    memory_dir='./memory',           # Markdown 存储目录
    vector_dir='./data/tfidf_memory', # 向量库目录
    collection_name='my_memory'       # 集合名称
)

print('✅ 记忆系统初始化完成')
"
```

### 4. 添加记忆

```bash
# 命令行方式
python3 scripts/memory.py add \
  --content "今天完成了记忆系统的架构设计" \
  --type insight \
  --emotion happy \
  --importance 8

# Python API 方式
from dual_memory import DualMemoryStore

store = DualMemoryStore()

# 添加普通记忆
store.add_memory(
    content="这是一个重要的洞察",
    memory_type="insight",
    emotion="excited",
    importance=9
)

# 添加认知材料（legacy）
store.add_memory(
    content="关于认知沉淀的核心观点",
    memory_type="insight",
    legacy=True,  # 标记为认知材料
    source="somebody"
)
```

### 5. 搜索记忆

```bash
# 命令行方式
python3 scripts/memory.py search --query "记忆系统" --n 5

# Python API 方式（自动记录使用情境）
results = store.search(
    query="记忆系统",
    n_results=5,
    current_context="技术讨论"  # 自动记录使用情境
)

# 按层级筛选
raw_results = store.search(query="规律", filter_layer="raw")
sediment_results = store.search(query="规律", filter_layer="sediment")
```

### 6. 查看统计

```bash
python3 scripts/memory.py stats
```

输出示例：
```json
{
  "total_count": 42,
  "layer_distribution": {
    "raw": 38,
    "sediment": 4
  },
  "legacy_count": 12,
  "total_domain_crossings": 15
}
```

---

## 📚 API 参考

### DualMemoryStore

```python
from dual_memory import DualMemoryStore

store = DualMemoryStore(
    memory_dir="./memory",
    vector_dir="./data/tfidf_memory",
    collection_name="my_memory"
)
```

#### 方法

| 方法 | 参数 | 返回 | 描述 |
|------|------|------|------|
| `add_memory()` | content, type, emotion... | dict | 添加记忆 |
| `search()` | query, n_results, current_context... | list | 搜索记忆 |
| `get_by_layer()` | layer, limit | list | 按层级获取 |
| `get_stats()` | - | dict | 获取统计 |
| `delete_memory()` | memory_id | bool | 删除记忆 |

### VectorMemoryStoreTFIDF（底层向量库）

```python
from vector_memory_tfidf import VectorMemoryStoreTFIDF, MemoryLayer

store = VectorMemoryStoreTFIDF(
    persist_directory="./data/tfidf_memory",
    collection_name="my_memory"
)
```

---

## 🎯 设计哲学

### 1. 理解即锚定
新信息和已有认知产生共鸣 → 自然沉淀。不是建索引，是让关联自己长出来。

### 2. 经验即索引
回忆不是"检索"，是"回到那个经历"。情境被动记录，不用预设标签。

### 3. 编织优于分类
不是分类柜，是关系网。两件无关的事，底层规律相同，就会被编织在一起。

### 4. 沉淀代表内化
跨域验证过的认知，才是真正属于自己的。3个不同情境被调用 = 自动沉淀。

### 5. 认知材料的开放态度
Legacy 不是既定结论，是认知材料，需自行消化确认。

---

## 🔧 高级配置

### 自定义沉淀阈值

```python
# 修改 vector_memory_tfidf.py 中的阈值
if memory["domain_crossings"] >= 3:  # 默认3，可调整
    memory["layer"] = MemoryLayer.SEDIMENT.value
```

### 添加遗忘机制

```python
# 可在 search 或定时任务中调用
def auto_archive(store, days_threshold=30):
    """自动归档久未使用的记忆"""
    memories = store.get_all()
    for m in memories:
        if (m.get("domain_crossings", 0) == 0 and
            m.get("importance", 5) < 5):
            store.vector_store.update_status(
                m["id"],
                new_status="archived"
            )
```

### 与 OpenClaw 集成

在 OpenClaw 的 `skills/memory-vector/SKILL.md` 中配置：

```yaml
# SKILL.md
name: memory-vector
description: 向量记忆存储系统，支持...
```

---

## 📂 目录结构

```
ai-memory-store/
├── dual_memory.py              # 双写记忆系统
├── vector_memory_tfidf.py      # TF-IDF 向量存储
├── scripts/
│   └── memory.py               # 命令行工具
├── memory/                     # Markdown 存储（自动创建）
├── data/
│   └── tfidf_memory/          # 向量库存储（自动创建）
└── README.md
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 开源协议

MIT License

---

## 🙏 致谢

这个项目的核心理念来自与XXX的一次深度对话：

> "两件风马牛不相及的事情，里面的规律是一样的，这两个情境被以这个规律为纽带关联了。"

---

*让 AI 从"工具"走向"同伴"，从"存储"走向"认知沉淀"。*
