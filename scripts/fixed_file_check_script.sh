#!/bin/bash

# 测试修复的文件检查逻辑
expected_files="rwfn.out rmix.out"

echo "原始字符串: $expected_files"

# 将文件列表转换为数组进行处理
files_array=($expected_files)
file_count=${#files_array[@]}

echo "预期文件数量: $file_count 个"
index=1
for file in "${files_array[@]}"; do
    echo "  [$index]: '$file'"
    index=$((index + 1))
done

echo "检查文件存在性:"
for file in "${files_array[@]}"; do
    echo "检查文件: '$file'"
    if [ -f "$file" ]; then
        echo "  ✅ 文件存在"
    else
        echo "  ❌ 文件不存在"
    fi
done 