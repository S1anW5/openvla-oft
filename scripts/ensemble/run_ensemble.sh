#!/bin/bash
# 在 tmux 中启动 ensemble 训练，3 个成员顺序执行
# 用法: bash run_ensemble.sh

SESSION="ensemble"
WORKDIR="/hdd/slwu/test_5_2/openvla-oft"

# 如果 session 已存在则先删除
tmux kill-session -t $SESSION 2>/dev/null

# 新建 tmux session
tmux new-session -d -s $SESSION -x 220 -y 50

# 发送训练命令
tmux send-keys -t $SESSION "conda activate openvla" Enter
tmux send-keys -t $SESSION "cd $WORKDIR" Enter
tmux send-keys -t $SESSION "mkdir -p runs/logs && bash train_ensemble.sh 2>&1 | tee runs/logs/ensemble_train.log" Enter

echo "训练已在 tmux session '${SESSION}' 中启动"
echo "查看进度: tmux attach -t ${SESSION}"
echo "后台运行: Ctrl+B D 可 detach"
echo "查看日志: tail -f ${WORKDIR}/runs/logs/ensemble_train.log"
