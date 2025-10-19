#!/bin/bash

# 定义变量
SRC_DIR="test1/1N"   # 源目录
DEST_DIR="验证N" # 目标目录
SOURCE_TAG="processed_1N"               # 来源标签（可自定义）

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 进入源目录并遍历所有文件
for file in "$SRC_DIR"/*; do
    # 判断是否为文件（排除子目录）
    if [[ -f "$file" ]]; then
        # 获取文件名（含路径）→ 文件基础名
        filename=$(basename "$file")            # 如 data.csv
        name="${filename%.*}"                   # 去掉扩展名，如 data
        ext="${filename##*.}"                   # 扩展名，如 csv

        # 构造新文件名：name_source_tag.ext
        if [[ "$ext" != "$filename" ]]; then
            # 有扩展名的情况
            new_filename="${name}_${SOURCE_TAG}.${ext}"
        else
            # 无扩展名的情况
            new_filename="${filename}_${SOURCE_TAG}"
        fi

        # 复制文件
        cp "$file" "$DEST_DIR/$new_filename"
        echo "Copied: $filename → $new_filename"
    fi
done