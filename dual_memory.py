#!/usr/bin/env python3
"""
双写记忆系统 - Markdown + 向量数据库

实现太玄需求：
1. 同时写入 Markdown 文件（人类可读）和向量库（语义搜索）
2. 支持从历史 Markdown 文件导入记忆
3. 统一的记忆管理 API

使用方式：
- from dual_memory import DualMemoryStore
- store = DualMemoryStore()
- store.add_memory(content, ...)  # 自动双写
- results = store.search(query)   # 向量搜索
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# 导入向量存储
from vector_memory_tfidf import VectorMemoryStoreTFIDF


class DualMemoryStore:
    """
    双写记忆系统
    
    同时写入：
    1. Markdown 文件 - 人类可读的日常记录
    2. 向量数据库 - 支持语义搜索
    
    特性：
    - 自动按日期创建 Markdown 文件
    - 支持元数据过滤搜索
    - 支持从历史文件导入记忆
    """
    
    def __init__(
        self,
        memory_dir: str = "/workspace/projects/workspace/memory",
        vector_dir: str = "/workspace/data/tfidf_memory",
        collection_name: str = "taixuan_memory"
    ):
        """
        初始化双写记忆系统
        
        Args:
            memory_dir: Markdown 记忆文件目录
            vector_dir: 向量数据库目录
            collection_name: 向量集合名称
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化向量存储
        self.vector_store = VectorMemoryStoreTFIDF(
            persist_directory=vector_dir,
            collection_name=collection_name
        )
        
        print(f"✅ 双写记忆系统初始化完成")
        print(f"   📁 Markdown 目录: {memory_dir}")
        print(f"   📊 向量库记录数: {self.vector_store.count()}")
    
    def _get_today_file(self) -> Path:
        """获取今天的记忆文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.memory_dir / f"{today}.md"
    
    def _format_memory_markdown(
        self,
        content: str,
        memory_type: str,
        emotion: str,
        importance: int,
        legacy: bool,
        source: str,
        context: str,
        tags: List[str],
        memory_layer: str = "raw"
    ) -> str:
        """格式化记忆为 Markdown"""
        timestamp = datetime.now().strftime("%H:%M")
        
        # 类型图标
        type_icons = {
            "conversation": "💬",
            "insight": "💡",
            "milestone": "🎯",
            "daily": "📝",
            "emotion": "❤️",
            "todo": "✅",
            "general": "📌"
        }
        icon = type_icons.get(memory_type, "📌")
        
        # 层级图标
        layer_icons = {
            "raw": "📝",
            "sediment": "🌊"
        }
        layer_icon = layer_icons.get(memory_layer, "📝")
        
        # 情感标签
        emotion_labels = {
            "happy": "😊 开心",
            "sad": "😢 难过",
            "excited": "🤩 兴奋",
            "neutral": "😐 平静",
            "proud": "🦚 自豪",
            "worried": "😟 担忧",
            "grateful": "🙏 感激",
            "angry": "😠 生气"
        }
        emotion_str = emotion_labels.get(emotion, emotion)
        
        # 构建条目
        lines = [
            f"#### {icon} {timestamp} - {memory_type.capitalize()}{' [LEGACY]' if legacy else ''} {layer_icon}",
            "",
            content,
            ""
        ]
        
        # 添加元数据
        meta_parts = []
        if source:
            meta_parts.append(f"来源: {source}")
        if context:
            meta_parts.append(f"上下文: {context}")
        if tags:
            meta_parts.append(f"标签: {', '.join(tags)}")
        meta_parts.append(f"情感: {emotion_str}")
        meta_parts.append(f"重要性: {'⭐' * min(importance, 10)}")
        meta_parts.append(f"层级: {memory_layer}")
        
        lines.append(" | ".join(meta_parts))
        lines.append("")
        lines.append("---")
        
        return "\n".join(lines)
    
    def _append_to_markdown(self, content: str) -> bool:
        """追加记忆到 Markdown 文件"""
        today_file = self._get_today_file()
        
        # 如果文件不存在，创建新文件
        if not today_file.exists():
            today = datetime.now().strftime("%Y-%m-%d")
            header = f"# {today} 日常记录\n\n"
            with open(today_file, 'w', encoding='utf-8') as f:
                f.write(header)
        
        # 追加记忆
        with open(today_file, 'a', encoding='utf-8') as f:
            f.write("\n" + content + "\n")
        
        return True
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        emotion: str = "neutral",
        importance: int = 5,
        legacy: bool = False,
        source: str = "",
        context: str = "",
        tags: List[str] = None,
        memory_layer: str = "raw"
    ) -> Dict[str, Any]:
        """
        添加记忆（双写）
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (conversation, insight, milestone, daily, emotion, todo, general)
            emotion: 情感标签
            importance: 重要性 (1-10)
            legacy: 是否为legacy，认知材料来源（非既定结论）
            source: 信息来源（如"马黎"、"豆包老师"、"我独立观察"）
            context: 上下文描述
            tags: 标签列表
            memory_layer: 记忆层级 (raw=原始记忆, sediment=认知沉淀)
            
        Returns:
            包含 memory_id 和状态的字典
        """
        # 1. 格式化并写入 Markdown
        markdown_content = self._format_memory_markdown(
            content=content,
            memory_type=memory_type,
            emotion=emotion,
            importance=importance,
            legacy=legacy,
            source=source,
            context=context,
            tags=tags or [],
            memory_layer=memory_layer
        )
        
        today_file = self._get_today_file()
        self._append_to_markdown(markdown_content)
        
        # 2. 写入向量数据库
        memory_id = self.vector_store.add_memory(
            content=content,
            memory_type=memory_type,
            emotion=emotion,
            importance=importance,
            legacy=legacy,
            source=source,
            context=context,
            source_file=str(today_file.name),
            tags=tags or [],
            memory_layer=memory_layer
        )
        
        layer_icon = "🌊" if memory_layer == "sediment" else "📝"
        flag = " [LEGACY]" if legacy else ""
        print(f"✅ 记忆已双写: {memory_id[:8]}... {layer_icon}{flag}")
        print(f"   📄 文件: {today_file.name}")
        
        return {
            "memory_id": memory_id,
            "source_file": str(today_file.name),
            "status": "success"
        }
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_type: Optional[str] = None,
        filter_emotion: Optional[str] = None,
        filter_min_importance: Optional[int] = None,
        filter_legacy: Optional[bool] = None,
        filter_layer: Optional[str] = None,
        current_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆（向量搜索）
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_type: 按类型过滤
            filter_emotion: 按情感过滤
            filter_min_importance: 最小重要性
            filter_legacy: 仅返回legacy记忆（True）或排除（False）
            filter_layer: 按层级过滤 (raw/sediment)
            current_context: 当前调用情境（用于记录使用场景）
            
        Returns:
            匹配的记忆列表
        """
        return self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_type=filter_type,
            filter_emotion=filter_emotion,
            filter_min_importance=filter_min_importance,
            filter_legacy=filter_legacy,
            filter_layer=filter_layer,
            current_context=current_context
        )
    
    def search_by_keywords(
        self,
        keywords: List[str],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        关键词搜索（精确匹配）
        
        Args:
            keywords: 关键词列表
            n_results: 返回结果数量
            
        Returns:
            包含关键词的记忆列表
        """
        return self.vector_store.search_by_keywords(
            keywords=keywords,
            n_results=n_results
        )
    
    def get_recent(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取最近的记忆
        
        Args:
            days: 天数
            limit: 最大数量
            
        Returns:
            最近的记忆列表
        """
        return self.vector_store.get_all(limit=limit)
    
    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        return self.vector_store.get_all(limit=limit)
    
    def get_by_layer(self, layer: str, limit: int = 100) -> List[Dict[str, Any]]:
        """按层级获取记忆
        
        Args:
            layer: 层级 (raw/sediment)
            limit: 限制数量
        """
        return self.vector_store.get_all(limit=limit, filter_layer=layer)
    
    def count(self) -> int:
        """获取记忆总数"""
        return self.vector_store.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.vector_store.get_stats()
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        return self.vector_store.delete_memory(memory_id)
    
    def clear_all(self) -> bool:
        """清空所有记忆"""
        return self.vector_store.clear_all()
    
    def import_from_markdown(self, file_path: str = None) -> int:
        """
        从 Markdown 文件导入历史记忆
        
        Args:
            file_path: 指定文件路径，如果为 None 则导入所有文件
            
        Returns:
            导入的记忆数量
        """
        imported_count = 0
        
        if file_path:
            files = [Path(file_path)]
        else:
            files = list(self.memory_dir.glob("*.md"))
        
        for md_file in files:
            if md_file.name == "heartbeat-state.json" or md_file.name == "todo.md":
                continue
            
            print(f"📂 导入文件: {md_file.name}")
            
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析日期
            date_match = re.match(r'# (\d{4}-\d{2}-\d{2})', content)
            date_str = date_match.group(1) if date_match else md_file.stem
            
            # 简单解析：将每个段落作为一条记忆
            sections = re.split(r'\n## ', content)
            
            for section in sections[1:]:  # 跳过标题
                lines = section.strip().split('\n')
                if len(lines) < 2:
                    continue
                
                # 提取标题作为类型
                title = lines[0].strip()
                memory_type = "general"
                if "心跳" in title:
                    memory_type = "daily"
                elif "讨论" in title:
                    memory_type = "conversation"
                elif "完成" in title or "里程碑" in title:
                    memory_type = "milestone"
                
                # 内容为剩余部分
                memory_content = '\n'.join(lines[1:]).strip()
                if len(memory_content) < 10:
                    continue
                
                # 添加到向量库
                self.vector_store.add_memory(
                    content=memory_content[:500],  # 限制长度
                    memory_type=memory_type,
                    emotion="neutral",
                    importance=5,
                    context=f"从 {date_str} 导入",
                    source_file=md_file.name,
                    tags=[]
                )
                imported_count += 1
        
        print(f"✅ 导入完成，共 {imported_count} 条记忆")
        return imported_count


def main():
    """测试主函数"""
    print("=" * 60)
    print("🧪 双写记忆系统测试")
    print("=" * 60)
    
    try:
        # 初始化
        print("\n📦 初始化双写记忆系统...")
        store = DualMemoryStore(
            memory_dir="/workspace/projects/workspace/memory",
            vector_dir="/workspace/data/tfidf_memory",
            collection_name="taixuan_memory"
        )
        
        print("\n📝 添加测试记忆（双写）...")
        
        # 添加记忆
        result = store.add_memory(
            content="今天和太玄一起讨论了向量数据库的集成方案，我们决定使用 TF-IDF + FAISS 方案，因为不需要下载外部模型。",
            memory_type="conversation",
            emotion="happy",
            importance=8,
            participants=["太玄", "扣子哥"],
            context="技术讨论",
            tags=["向量数据库", "TF-IDF", "FAISS", "双写"]
        )
        
        print(f"   记忆 ID: {result['memory_id'][:8]}...")
        print(f"   源文件: {result['source_file']}")
        
        # 添加更多记忆
        store.add_memory(
            content="太玄提出了一个很好的想法：将记忆同时写入 Markdown 文件和向量数据库，实现人类可读和机器可搜索的双赢。",
            memory_type="insight",
            emotion="excited",
            importance=9,
            participants=["太玄"],
            context="需求讨论",
            tags=["双写", "记忆系统", "创新"]
        )
        
        print(f"\n📊 当前记忆总数: {store.count()}")
        
        # 搜索测试
        print("\n🔍 搜索测试: '向量数据库'")
        results = store.search("向量数据库", n_results=3)
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] {r['content'][:60]}...")
            print(f"      类型: {r.get('type')} | 重要性: {r.get('importance')}")
        
        # 检查 Markdown 文件
        print("\n📄 检查今天的 Markdown 文件:")
        today_file = store._get_today_file()
        if today_file.exists():
            with open(today_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(content[:500])
        
        print("\n" + "=" * 60)
        print("✅ 双写记忆系统测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
