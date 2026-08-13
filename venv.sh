#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

usage() {
    echo "用法: $0 {create|activate|deactivate}"
    echo ""
    echo "  create      创建虚拟环境 (若已存在则跳过)"
    echo "  activate    激活虚拟环境 (在当前 shell 中生效)"
    echo "  deactivate  取消激活虚拟环境"
    exit 1
}

cmd_create() {
    if [ -d "$VENV_DIR" ]; then
        echo "✅ 虚拟环境已存在: $VENV_DIR"
    else
        echo "🔨 正在创建虚拟环境: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "✅ 创建完成"
    fi
}

cmd_activate() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "❌ 虚拟环境不存在，请先执行: $0 create"
        exit 1
    fi
    # 注意：source 必须在当前 shell 执行才能生效
    source "$VENV_DIR/bin/activate"
    echo "✅ 已激活虚拟环境: $(which python)"
}

cmd_deactivate() {
    if command -v deactivate &>/dev/null; then
        deactivate
        echo "✅ 已取消激活虚拟环境"
    else
        echo "⚠️  当前没有激活的虚拟环境"
    fi
}

# ---- 主入口 ----
[ $# -lt 1 ] && usage

case "$1" in
    create)     cmd_create ;;
    activate)   cmd_activate ;;
    deactivate) cmd_deactivate ;;
    *)          usage ;;
esac