'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 18:18:46
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:18:50
'''
import os

EXCLUDE_DIRS = {".venv", ".vscode", ".git"}  # 要排除的文件夹

def print_tree(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# 使用示例
print_tree(".")
