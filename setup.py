#!/usr/bin/env python3
"""
安装脚本：将记忆系统安装为 Python 包
"""

from setuptools import setup, find_packages

setup(
    name="ai-memory-store",
    version="1.0.0",
    description="AI Memory Store - 让 AI 拥有认知沉淀能力",
    author="Your Name",
    author_email="your@email.com",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "faiss-cpu",
    ],
    python_requires=">=3.8",
)
