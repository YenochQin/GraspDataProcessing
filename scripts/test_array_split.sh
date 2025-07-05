#!/bin/zsh
# 测试数组分割在zsh下的行为

echo "Shell: $SHELL"
echo "ZSH版本: ${ZSH_VERSION:-未检测到}"
echo "BASH版本: ${BASH_VERSION:-未检测到}"
echo ""

# 测试字符串
expected_files="rwfn.out rmix.out"
echo "测试字符串: '$expected_files'"
echo ""

# 方法1: 直接分割
echo "方法1: 直接分割"
files_array1=($expected_files)
echo "数组元素数量: ${#files_array1[@]}"
echo "元素1: '${files_array1[1]}'"
echo "元素2: '${files_array1[2]}'"
echo ""

# 方法2: 使用 ${=var}
echo "方法2: 使用 \${=var}"
files_array2=(${=expected_files})
echo "数组元素数量: ${#files_array2[@]}"
echo "元素1: '${files_array2[1]}'"
echo "元素2: '${files_array2[2]}'"
echo ""

# 方法3: 使用 read
echo "方法3: 使用 read -A (zsh)"
files_array3=()
read -A files_array3 <<< "$expected_files"
echo "数组元素数量: ${#files_array3[@]}"
echo "元素1: '${files_array3[1]}'"
echo "元素2: '${files_array3[2]}'"
echo ""

# 方法4: 使用通用方法
echo "方法4: 通用方法"
files_array4=()
if [[ -n "${ZSH_VERSION:-}" ]]; then
    echo "检测到 zsh，使用 \${=var}"
    files_array4=(${=expected_files})
else
    echo "非 zsh，直接分割"
    files_array4=($expected_files)
fi
echo "数组元素数量: ${#files_array4[@]}"
echo "元素1: '${files_array4[1]}'"
echo "元素2: '${files_array4[2]}'"
echo ""

# 测试文件检查
echo "测试文件检查逻辑:"
for file in "${files_array4[@]}"; do
    echo "检查文件: '$file'"
    if [ -f "$file" ]; then
        echo "  - 文件存在"
    else
        echo "  - 文件不存在"
    fi
done 