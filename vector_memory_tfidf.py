#!/usr/bin/env python3
"""
向量记忆存储 - TF-IDF + FAISS 版本（带状态管理）

使用 TF-IDF 向量化 + FAISS 索引的混合方案：
- 无需下载任何外部模型
- 使用 scikit-learn 的 TF-IDF 向量化器
- 使用 FAISS 进行高效向量搜索
- 支持中英文混合文本
- 支持记忆状态管理（active/deprecated/uncertain/archived）

数据存储结构：
- /workspace/data/tfidf_memory/index.faiss - FAISS 向量索引
- /workspace/data/tfidf_memory/metadata.json - 元数据存储
- /workspace/data/tfidf_memory/vectorizer.pkl - TF-IDF 向量化器
"""

import os
import json
import uuid
import pickle
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np

# FAISS 向量搜索
import faiss

# TF-IDF 向量化
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MemoryStatus(str, Enum):
    """记忆状态枚举"""
    ACTIVE = "active"        # 当前有效
    DEPRECATED = "deprecated" # 已过期/被替代
    UNCERTAIN = "uncertain"   # 不确定/待验证
    ARCHIVED = "archived"     # 归档（不参与搜索）


class MemoryLayer(str, Enum):
    """记忆层级枚举"""
    RAW = "raw"           # 原始记忆：未经内化的初始记录
    SEDIMENT = "sediment" # 认知沉淀：跨域确认过的规律


class VectorMemoryStoreTFIDF:
    """
    基于 TF-IDF + FAISS 的向量记忆存储
    
    特性：
    - 无需下载任何模型
    - 支持中英文混合文本
    - 持久化存储
    - 关键词搜索和相似度匹配
    - 元数据过滤（类型、情感、重要性）
    - 记忆状态管理（active/deprecated/uncertain/archived）
    """
    
    def __init__(
        self,
        persist_directory: str = "/workspace/data/tfidf_memory",
        collection_name: str = "taixuan_memory"
    ):
        """
        初始化向量记忆存储
        
        Args:
            persist_directory: 持久化存储目录
            collection_name: 集合名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 文件路径
        self.index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
        self.metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.json")
        self.vectorizer_path = os.path.join(persist_directory, f"{collection_name}_vectorizer.pkl")
        self.documents_path = os.path.join(persist_directory, f"{collection_name}_documents.json")
        
        # 初始化 TF-IDF 向量化器（支持中英文）
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=5000,  # 限制特征数量
            ngram_range=(1, 2),  # 使用 1-gram 和 2-gram
            min_df=1,
            max_df=1.0,  # 不忽略任何词（适合小数据集）
            # 中文分词支持
            token_pattern=r'(?u)\b\w+\b|[^\s\w]'  # 匹配单词和单个字符
        )
        
        # 存储文档列表（用于重新训练向量化器）
        self.documents_list = []
        
        # 元数据存储
        self.metadata_store = {}
        
        # 索引
        self.index = None
        self.embedding_dimension = None
        
        # 加载现有数据
        self._load_or_initialize()
        
        print(f"✅ 向量记忆存储初始化完成")
        print(f"   📁 持久化目录: {persist_directory}")
        print(f"   📦 集合名称: {collection_name}")
        print(f"   📊 现有记录数: {len(self.metadata_store)}")
    
    def _migrate_fields(self):
        """迁移：为旧记忆补上新字段"""
        migrated = 0
        for memory_id, memory in self.metadata_store.items():
            # 补状态字段
            if "status" not in memory:
                memory["status"] = MemoryStatus.ACTIVE.value
                memory["status_history"] = []
                memory["replaced_by"] = None
                memory["replaced_reason"] = None
                migrated += 1
            
            # 补记忆层级字段
            if "layer" not in memory:
                memory["layer"] = MemoryLayer.RAW.value
            
            # 补情境记录字段
            if "usage_contexts" not in memory:
                memory["usage_contexts"] = []
            
            # 补跨域计数字段
            if "domain_crossings" not in memory:
                memory["domain_crossings"] = 0
        
        if migrated > 0:
            print(f"🔄 迁移了 {migrated} 条旧记忆，添加新字段")
            self._save_metadata()
    
    def _load_or_initialize(self):
        """加载或初始化索引"""
        # 加载元数据
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_store = json.load(f)
            
            # 迁移：为旧记忆补上新字段
            self._migrate_fields()
        
        # 加载文档列表
        if os.path.exists(self.documents_path):
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                self.documents_list = json.load(f)
        
        # 加载向量化器和索引
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.index_path):
            print(f"📂 加载现有向量化器和索引...")
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.index = faiss.read_index(self.index_path)
            print(f"✅ 加载完成，包含 {self.index.ntotal} 个向量")
        else:
            print(f"🆕 创建新的索引...")
            print(f"✅ 索引将在添加第一个文档时创建")
    
    def _save_metadata(self):
        """保存元数据（单独保存，不重建索引）"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
    
    def _save_all(self):
        """保存所有数据"""
        # 保存索引
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        # 保存元数据
        self._save_metadata()
        
        # 保存文档列表
        with open(self.documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents_list, f, ensure_ascii=False, indent=2)
        
        # 保存向量化器
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def _rebuild_index(self):
        """重建 TF-IDF 向量化器和 FAISS 索引"""
        if not self.documents_list:
            self.index = None
            return
        
        # 重新训练向量化器
        tfidf_matrix = self.vectorizer.fit_transform(self.documents_list)
        
        # 获取嵌入维度
        self.embedding_dimension = tfidf_matrix.shape[1]
        
        # 创建 FAISS 索引
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # 添加所有向量
        vectors = tfidf_matrix.toarray().astype('float32')
        self.index.add(vectors)
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        emotion: str = "neutral",
        importance: int = 5,
        legacy: bool = False,
        source: str = "",
        context: str = "",
        source_file: str = "",
        tags: List[str] = None,
        status: str = MemoryStatus.ACTIVE.value,
        memory_layer: str = MemoryLayer.RAW.value
    ) -> str:
        """
        添加一条记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (conversation, insight, milestone, etc.)
            emotion: 情感标签 (happy, sad, excited, neutral, etc.)
            importance: 重要性 (1-10)，主观重要程度
            legacy: 是否为legacy，认知材料来源（非既定结论）
            source: 信息来源（如"马黎"、"豆包老师"、"我独立观察"）
            context: 上下文描述
            source_file: 来源文件
            tags: 标签列表
            status: 状态 (active, deprecated, uncertain, archived)
            memory_layer: 记忆层级 (raw=原始记忆, sediment=认知沉淀)
            
        Returns:
            记忆 ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 添加到文档列表
        doc_index = len(self.documents_list)
        self.documents_list.append(content)
        
        # 保存元数据
        metadata = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "emotion": emotion,
            "importance": importance,
            "legacy": legacy,
            "source": source,
            "context": context,
            "source_file": source_file,
            "timestamp": timestamp,
            "tags": tags or [],
            "doc_index": doc_index,
            # 记忆层级
            "layer": memory_layer,
            # 情境记录（被调用时记录）
            "usage_contexts": [],
            # 跨域计数（同一条规律在多少不同领域被使用）
            "domain_crossings": 0,
            # 状态管理字段
            "status": status,
            "status_history": [
                {
                    "status": status,
                    "timestamp": timestamp,
                    "context": "初始创建"
                }
            ],
            "replaced_by": None,
            "replaced_reason": None
        }
        
        self.metadata_store[memory_id] = metadata
        
        # 重建索引
        self._rebuild_index()
        
        # 保存
        self._save_all()
        
        layer_icon = "🌊" if memory_layer == MemoryLayer.SEDIMENT.value else "📝"
        flag = " [LEGACY]" if legacy else ""
        print(f"✅ 已添加记忆: {memory_id[:8]}... {layer_icon} ({memory_layer})")
        if legacy:
            print(f"   ⚠️ Legacy 认知材料，需自行消化确认")
        return memory_id
    
    def _record_usage_context(self, memory_id: str, context: str) -> None:
        """
        记录记忆被调用时的情境
        
        Args:
            memory_id: 记忆 ID
            context: 当前使用情境描述
        """
        if memory_id not in self.metadata_store:
            return
        
        memory = self.metadata_store[memory_id]
        timestamp = datetime.now().isoformat()
        
        # 初始化字段
        if "usage_contexts" not in memory:
            memory["usage_contexts"] = []
        if "domain_crossings" not in memory:
            memory["domain_crossings"] = 0
        
        # 检查是否已在相同情境中使用
        existing_contexts = [uc.get("context") for uc in memory["usage_contexts"]]
        if context not in existing_contexts:
            # 新情境 → 增加跨域计数
            memory["usage_contexts"].append({
                "context": context,
                "timestamp": timestamp
            })
            memory["domain_crossings"] += 1
            
            # 检查是否需要沉淀（跨域阈值：3）
            if memory["domain_crossings"] >= 3 and memory.get("layer") == MemoryLayer.RAW.value:
                old_layer = memory.get("layer")
                memory["layer"] = MemoryLayer.SEDIMENT.value
                print(f"🌊 记忆 {memory_id[:8]}... 跨域沉淀: {old_layer} → {memory['layer']} (domain_crossings: {memory['domain_crossings']})")
        
        # 更新最后访问时间
        memory["last_accessed"] = timestamp
        self._save_metadata()
    
    def update_status(
        self,
        memory_id: str,
        new_status: str,
        replaced_by: str = None,
        reason: str = None,
        context: str = ""
    ) -> bool:
        """
        更新记忆状态
        
        Args:
            memory_id: 记忆 ID
            new_status: 新状态 (active, deprecated, uncertain, archived)
            replaced_by: 替代该记忆的新记忆 ID
            reason: 状态变更原因
            context: 变更上下文
            
        Returns:
            是否更新成功
        """
        if memory_id not in self.metadata_store:
            print(f"❌ 记忆不存在: {memory_id[:8]}...")
            return False
        
        memory = self.metadata_store[memory_id]
        old_status = memory.get("status", MemoryStatus.ACTIVE.value)
        timestamp = datetime.now().isoformat()
        
        # 更新状态
        memory["status"] = new_status
        
        # 记录状态历史
        history_entry = {
            "from": old_status,
            "to": new_status,
            "timestamp": timestamp,
            "context": context,
            "reason": reason
        }
        if "status_history" not in memory:
            memory["status_history"] = []
        memory["status_history"].append(history_entry)
        
        # 如果被替代，记录替代关系
        if replaced_by:
            memory["replaced_by"] = replaced_by
        if reason:
            memory["replaced_reason"] = reason
        
        # 保存
        self._save_metadata()
        
        print(f"✅ 已更新记忆状态: {memory_id[:8]}... ({old_status} → {new_status})")
        if reason:
            print(f"   原因: {reason}")
        
        return True
    
    def get_status(self, memory_id: str) -> Optional[str]:
        """获取记忆状态"""
        if memory_id not in self.metadata_store:
            return None
        return self.metadata_store[memory_id].get("status")
    
    def get_status_history(self, memory_id: str) -> List[Dict]:
        """获取记忆的状态变更历史"""
        if memory_id not in self.metadata_store:
            return []
        return self.metadata_store[memory_id].get("status_history", [])
    
    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取记忆"""
        return self.metadata_store.get(memory_id)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_type: Optional[str] = None,
        filter_emotion: Optional[str] = None,
        filter_min_importance: Optional[int] = None,
        filter_legacy: Optional[bool] = None,
        filter_status: Optional[List[str]] = None,
        filter_layer: Optional[str] = None,
        include_deprecated: bool = False,
        current_context: str = None
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_type: 按类型过滤
            filter_emotion: 按情感过滤
            filter_min_importance: 最小重要性过滤
            filter_legacy: 过滤legacy记忆（True=仅legacy, False=排除legacy, None=不过滤）
            filter_status: 按状态过滤（默认为 ["active"]）
            filter_layer: 按层级过滤 (raw/sediment)
            include_deprecated: 是否包含 deprecated 记忆（默认 False）
            current_context: 当前调用情境（用于记录使用场景）
            
        Returns:
            匹配的记忆列表（按相似度排序）
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 默认状态过滤：只搜索 active 记忆
        if filter_status is None and not include_deprecated:
            filter_status = [MemoryStatus.ACTIVE.value]
        
        # 将查询转换为 TF-IDF 向量
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # 搜索更多结果，以便过滤后仍有足够数量
        k = min(n_results * 5, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # 获取所有记忆 ID
        memory_ids = list(self.metadata_store.keys())
        
        # 收集结果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(memory_ids):
                continue
            
            memory_id = memory_ids[idx]
            memory = self.metadata_store[memory_id].copy()
            
            # 应用状态过滤器
            if filter_status and memory.get("status") not in filter_status:
                continue
            
            # 应用其他过滤器
            if filter_type and memory.get("type") != filter_type:
                continue
            if filter_emotion and memory.get("emotion") != filter_emotion:
                continue
            if filter_min_importance and memory.get("importance", 0) < filter_min_importance:
                continue
            if filter_legacy is not None and memory.get("legacy", False) != filter_legacy:
                continue
            if filter_layer and memory.get("layer") != filter_layer:
                continue
            
            memory["distance"] = float(dist)
            # 计算相似度分数（距离越小越相似）
            memory["score"] = 1.0 / (1.0 + float(dist))
            results.append(memory)
            
            # 记录使用情境（用于跨域追踪）
            if current_context:
                self._record_usage_context(memory_id, current_context)
            
            if len(results) >= n_results:
                break
        
        return results
    
    def search_by_keywords(
        self,
        keywords: List[str],
        n_results: int = 5,
        include_deprecated: bool = False
    ) -> List[Dict[str, Any]]:
        """
        关键词搜索（精确匹配）
        
        Args:
            keywords: 关键词列表
            n_results: 返回结果数量
            include_deprecated: 是否包含 deprecated 记忆
            
        Returns:
            包含关键词的记忆列表
        """
        results = []
        for memory_id, memory in self.metadata_store.items():
            # 默认过滤 deprecated
            if not include_deprecated and memory.get("status") == MemoryStatus.DEPRECATED.value:
                continue
            
            content = memory.get("content", "").lower()
            tags = [t.lower() for t in memory.get("tags", [])]
            
            # 检查是否有任何关键词匹配
            matched = False
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in content or kw_lower in tags:
                    matched = True
                    break
            
            if matched:
                results.append(memory.copy())
        
        # 按重要性排序
        results.sort(key=lambda x: x.get("importance", 5), reverse=True)
        return results[:n_results]
    
    def get_all(
        self,
        limit: int = 100,
        include_deprecated: bool = False,
        filter_status: List[str] = None,
        filter_layer: str = None,
        filter_legacy: bool = None
    ) -> List[Dict[str, Any]]:
        """
        获取所有记忆
        
        Args:
            limit: 限制数量
            include_deprecated: 是否包含 deprecated 记忆
            filter_status: 按状态过滤
            filter_layer: 按层级过滤 (raw/sediment)
            filter_legacy: 仅返回legacy记忆（True）或排除legacy（False）
        """
        memories = list(self.metadata_store.values())
        
        # 应用状态过滤
        if filter_status:
            memories = [m for m in memories if m.get("status") in filter_status]
        elif not include_deprecated:
            memories = [m for m in memories if m.get("status") != MemoryStatus.DEPRECATED.value]
        
        # 层级过滤
        if filter_layer:
            memories = [m for m in memories if m.get("layer") == filter_layer]
        
        # legacy 过滤
        if filter_legacy is not None:
            memories = [m for m in memories if m.get("legacy", False) == filter_legacy]
        
        # 按时间戳排序（最新的在前）
        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return memories[:limit]
    
    def get_deprecated(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有 deprecated 记忆"""
        memories = [
            m.copy() for m in self.metadata_store.values()
            if m.get("status") == MemoryStatus.DEPRECATED.value
        ]
        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return memories[:limit]
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除一条记忆"""
        if memory_id not in self.metadata_store:
            return False
        
        # 获取文档索引
        doc_index = self.metadata_store[memory_id].get("doc_index")
        
        # 从元数据中删除
        del self.metadata_store[memory_id]
        
        # 从文档列表中删除
        if doc_index is not None and doc_index < len(self.documents_list):
            self.documents_list.pop(doc_index)
            # 更新其他记忆的 doc_index
            for mid, m in self.metadata_store.items():
                if m.get("doc_index", 0) > doc_index:
                    m["doc_index"] -= 1
        
        # 重建索引
        self._rebuild_index()
        
        # 保存
        self._save_all()
        
        print(f"✅ 已删除记忆: {memory_id[:8]}...")
        return True
    
    def clear_all(self) -> bool:
        """清空所有记忆"""
        self.index = None
        self.metadata_store = {}
        self.documents_list = []
        
        # 重新初始化向量化器
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b\w+\b|[^\s\w]'
        )
        
        # 保存
        self._save_all()
        
        print(f"✅ 已清空所有记忆")
        return True
    
    def count(self, include_deprecated: bool = False) -> int:
        """
        获取记忆数量
        
        Args:
            include_deprecated: 是否包含 deprecated 记忆
        """
        if include_deprecated:
            return len(self.metadata_store)
        return len([m for m in self.metadata_store.values() if m.get("status") != MemoryStatus.DEPRECATED.value])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        memories = list(self.metadata_store.values())
        
        # 统计各类型数量
        type_counts = {}
        emotion_counts = {}
        status_counts = {}
        layer_counts = {}
        importance_sum = 0
        legacy_count = 0
        legacy_sediment_count = 0
        all_tags = []
        total_domain_crossings = 0
        
        for m in memories:
            t = m.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            
            e = m.get("emotion", "neutral")
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            s = m.get("status", MemoryStatus.ACTIVE.value)
            status_counts[s] = status_counts.get(s, 0) + 1
            
            # 层级分布
            l = m.get("layer", MemoryLayer.RAW.value)
            layer_counts[l] = layer_counts.get(l, 0) + 1
            
            importance_sum += m.get("importance", 5)
            if m.get("legacy", False):
                legacy_count += 1
                if m.get("layer") == MemoryLayer.SEDIMENT.value:
                    legacy_sediment_count += 1
            all_tags.extend(m.get("tags", []))
            total_domain_crossings += m.get("domain_crossings", 0)
        
        # 统计标签频率
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_count": len(memories),
            "active_count": status_counts.get(MemoryStatus.ACTIVE.value, 0),
            "deprecated_count": status_counts.get(MemoryStatus.DEPRECATED.value, 0),
            "uncertain_count": status_counts.get(MemoryStatus.UNCERTAIN.value, 0),
            "archived_count": status_counts.get(MemoryStatus.ARCHIVED.value, 0),
            "legacy_count": legacy_count,
            "legacy_sediment_count": legacy_sediment_count,
            "type_distribution": type_counts,
            "emotion_distribution": emotion_counts,
            "status_distribution": status_counts,
            "layer_distribution": layer_counts,
            "tag_distribution": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
            "average_importance": importance_sum / len(memories) if memories else 0,
            "total_domain_crossings": total_domain_crossings,
            "index_size": self.index.ntotal if self.index else 0
        }


def main():
    """测试主函数"""
    print("=" * 60)
    print("🧪 向量记忆存储测试 (TF-IDF + FAISS) - 带状态管理")
    print("=" * 60)
    
    try:
        # 初始化存储
        print("\n📦 初始化向量存储...")
        store = VectorMemoryStoreTFIDF(
            persist_directory="/workspace/data/tfidf_memory",
            collection_name="taixuan_memory_test"
        )
        
        print("\n📝 添加测试记忆...")
        
        # 添加测试记忆
        id1 = store.add_memory(
            content="今天和太玄一起讨论了向量数据库的集成方案，我们决定使用 FAISS 作为解决方案。",
            memory_type="conversation",
            emotion="happy",
            importance=8,
            participants=["太玄", "扣子哥"],
            context="技术讨论",
            tags=["向量数据库", "FAISS", "集成"]
        )
        
        id2 = store.add_memory(
            content="太玄分享了他对 AI 助手记忆系统的想法，希望实现语义搜索和情感追踪功能。",
            memory_type="insight",
            emotion="excited",
            importance=9,
            participants=["太玄"],
            context="需求讨论",
            tags=["AI", "记忆系统", "语义搜索"]
        )
        
        id3 = store.add_memory(
            content="完成了 TF-IDF 向量数据库的集成测试，支持持久化存储和关键词搜索功能。",
            memory_type="milestone",
            emotion="proud",
            importance=7,
            participants=["扣子哥"],
            context="开发进展",
            tags=["TF-IDF", "FAISS", "测试", "集成"]
        )
        
        id4 = store.add_memory(
            content="跑步适合作为日常锻炼方式，对心肺功能有益。",
            memory_type="insight",
            emotion="neutral",
            importance=6,
            participants=["太玄"],
            context="健身讨论",
            tags=["健身", "跑步"]
        )
        
        print(f"\n📊 当前记忆数量: {store.count()}")
        
        # 测试状态更新
        print("\n🔄 测试状态更新...")
        success = store.update_status(
            id4,
            new_status=MemoryStatus.DEPRECATED.value,
            reason="跑步不适合大体重人群，膝盖会疼",
            replaced_by=id2,
            context="减重健身讨论"
        )
        print(f"   更新结果: {success}")
        
        # 查看状态历史
        print("\n📜 查看状态历史...")
        history = store.get_status_history(id4)
        print(f"   历史记录数: {len(history)}")
        for h in history:
            print(f"   - {h}")
        
        # 测试搜索（默认不包含 deprecated）
        print("\n🔍 测试搜索 (默认不包含 deprecated)...")
        results = store.search("健身", n_results=5)
        print(f"   找到 {len(results)} 条结果（应不包含跑步那条）")
        for r in results:
            print(f"   - {r['content'][:40]}... [status: {r.get('status')}]")
        
        # 测试搜索（包含 deprecated）
        print("\n🔍 测试搜索 (包含 deprecated)...")
        results = store.search("健身", n_results=5, include_deprecated=True)
        print(f"   找到 {len(results)} 条结果（应包含跑步那条）")
        for r in results:
            print(f"   - {r['content'][:40]}... [status: {r.get('status')}]")
        
        # 打印统计信息
        print("\n📈 统计信息:")
        stats = store.get_stats()
        print(f"   总数: {stats['total_count']}")
        print(f"   有效: {stats['active_count']}")
        print(f"   已过期: {stats['deprecated_count']}")
        print(f"   类型分布: {stats['type_distribution']}")
        print(f"   状态分布: {stats['status_distribution']}")
        
        # 测试持久化
        print("\n💾 测试持久化...")
        print("   重新加载存储...")
        store2 = VectorMemoryStoreTFIDF(
            persist_directory="/workspace/data/tfidf_memory",
            collection_name="taixuan_memory_test"
        )
        print(f"   ✅ 重新加载成功，记录数: {store2.count()}")
        print(f"   ✅ 状态保留: {store2.get_status(id4)}")
        
        print("\n" + "=" * 60)
        print("✅ 向量记忆存储测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
