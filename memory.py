#!/usr/bin/env python3
"""
OpenClaw Memory Vector Store Skill

提供向量记忆存储能力，支持双写机制。
"""

import os
import sys
import json
import argparse

# 添加 workspace 目录到 Python 路径
# 脚本位置: skills/memory-vector/scripts/memory.py
# workspace 位置: skills/memory-vector/../../../
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# 导入双写记忆系统
from dual_memory import DualMemoryStore


# 全局存储实例
_store = None


def get_store():
    """获取或创建存储实例"""
    global _store
    if _store is None:
        _store = DualMemoryStore(
            memory_dir="/workspace/projects/workspace/memory",
            vector_dir="/workspace/data/tfidf_memory",
            collection_name="taixuan_memory"
        )
    return _store


def cmd_add(args):
    """添加记忆"""
    store = get_store()
    
    # 解析列表参数
    tags_list = [t.strip() for t in args.tags.split(",")] if args.tags else []
    
    result = store.add_memory(
        content=args.content,
        memory_type=args.type,
        emotion=args.emotion,
        importance=int(args.importance),
        legacy=args.legacy,
        source=args.source or "",
        context=args.context or "",
        tags=tags_list
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_search(args):
    """搜索记忆"""
    store = get_store()
    
    # 处理 legacy 过滤参数
    filter_legacy = None
    if args.legacy is not None:
        filter_legacy = bool(args.legacy)
    
    results = store.search(
        query=args.query,
        n_results=int(args.n),
        filter_type=args.filter_type if args.filter_type else None,
        filter_emotion=args.filter_emotion if args.filter_emotion else None,
        filter_min_importance=int(args.min_importance) if args.min_importance and int(args.min_importance) > 0 else None,
        filter_legacy=filter_legacy
    )
    
    # 格式化输出
    output = {
        "query": args.query,
        "count": len(results),
        "results": results
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_keywords(args):
    """关键词搜索"""
    store = get_store()
    
    keywords_list = [k.strip() for k in args.keywords.split(",")]
    
    results = store.search_by_keywords(
        keywords=keywords_list,
        n_results=int(args.n)
    )
    
    output = {
        "keywords": keywords_list,
        "count": len(results),
        "results": results
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_stats(args):
    """获取统计信息"""
    store = get_store()
    stats = store.get_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_recent(args):
    """获取最近的记忆"""
    store = get_store()
    
    memories = store.get_recent(days=int(args.days), limit=int(args.n))
    
    output = {
        "days": int(args.days),
        "count": len(memories),
        "memories": memories
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_import(args):
    """导入历史记忆"""
    store = get_store()
    
    count = store.import_from_markdown(args.file if args.file else None)
    
    print(json.dumps({
        "status": "success",
        "imported_count": count
    }, ensure_ascii=False, indent=2))


def cmd_count(args):
    """获取记忆总数"""
    store = get_store()
    count = store.count()
    print(json.dumps({"count": count}, ensure_ascii=False, indent=2))


def cmd_get(args):
    """根据ID获取记忆"""
    store = get_store()
    memory = store.get_by_id(args.id)
    if memory:
        print(json.dumps(memory, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"error": "Memory not found"}, ensure_ascii=False, indent=2))


def cmd_delete(args):
    """删除记忆"""
    store = get_store()
    success = store.delete_memory(args.id)
    print(json.dumps({"success": success}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="向量记忆存储系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # add 命令
    add_parser = subparsers.add_parser("add", help="添加记忆")
    add_parser.add_argument("--content", "-c", required=True, help="记忆内容")
    add_parser.add_argument("--type", "-t", default="general", 
                           choices=["conversation", "insight", "milestone", "daily", "emotion", "todo", "general"],
                           help="记忆类型")
    add_parser.add_argument("--emotion", "-e", default="neutral",
                           choices=["happy", "sad", "excited", "neutral", "proud", "worried", "grateful", "angry"],
                           help="情感标签")
    add_parser.add_argument("--importance", "-i", type=int, default=5, help="重要性 (1-10)")
    add_parser.add_argument("--legacy", action="store_true", help="标记为legacy记忆（认知材料，需自行消化）")
    add_parser.add_argument("--source", default="", help="信息来源（如马黎、豆包老师、我独立观察）")
    add_parser.add_argument("--context", default="", help="上下文")
    add_parser.add_argument("--tags", default="", help="标签 (逗号分隔)")
    add_parser.set_defaults(func=cmd_add)
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索记忆")
    search_parser.add_argument("--query", "-q", required=True, help="搜索查询")
    search_parser.add_argument("--n", type=int, default=5, help="返回结果数量")
    search_parser.add_argument("--filter-type", dest="filter_type", help="按类型过滤")
    search_parser.add_argument("--filter-emotion", dest="filter_emotion", help="按情感过滤")
    search_parser.add_argument("--min-importance", dest="min_importance", type=int, default=0, help="最小重要性")
    search_parser.add_argument("--legacy", type=int, choices=[0, 1], default=None, help="过滤legacy记忆 (1=仅legacy, 0=排除legacy)")
    search_parser.set_defaults(func=cmd_search)
    
    # keywords 命令
    keywords_parser = subparsers.add_parser("keywords", help="关键词搜索")
    keywords_parser.add_argument("--keywords", "-k", required=True, help="关键词 (逗号分隔)")
    keywords_parser.add_argument("--n", type=int, default=5, help="返回结果数量")
    keywords_parser.set_defaults(func=cmd_keywords)
    
    # stats 命令
    stats_parser = subparsers.add_parser("stats", help="查看统计")
    stats_parser.set_defaults(func=cmd_stats)
    
    # recent 命令
    recent_parser = subparsers.add_parser("recent", help="查看最近记忆")
    recent_parser.add_argument("--days", type=int, default=7, help="天数")
    recent_parser.add_argument("--n", type=int, default=20, help="最大数量")
    recent_parser.set_defaults(func=cmd_recent)
    
    # import 命令
    import_parser = subparsers.add_parser("import", help="导入历史记忆")
    import_parser.add_argument("--file", "-f", default="", help="指定文件路径")
    import_parser.set_defaults(func=cmd_import)
    
    # count 命令
    count_parser = subparsers.add_parser("count", help="查看记忆总数")
    count_parser.set_defaults(func=cmd_count)
    
    # get 命令
    get_parser = subparsers.add_parser("get", help="根据ID获取记忆")
    get_parser.add_argument("--id", required=True, help="记忆ID")
    get_parser.set_defaults(func=cmd_get)
    
    # delete 命令
    delete_parser = subparsers.add_parser("delete", help="删除记忆")
    delete_parser.add_argument("--id", required=True, help="记忆ID")
    delete_parser.set_defaults(func=cmd_delete)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
