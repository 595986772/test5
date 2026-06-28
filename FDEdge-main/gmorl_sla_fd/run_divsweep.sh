#!/usr/bin/env bash
# div_target 甜点扫描 (60ep 快代理 + 诊断): 找"打破塌缩但不杀能耗稀疏"的值。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
SCRATCH="/c/Users/陈亮/AppData/Local/Temp/claude/D--python-project----5/cc14f2bb-1e73-40ce-9aff-f5ab40b725d6/scratchpad"
for V in 002 003; do
  case $V in 002) T=0.02;; 003) T=0.03;; esac
  TAG=_val_div${V}
  echo "######## TRAIN $TAG (div_target=$T) ########"
  python train_frac_seq.py --env hetero --actor diffusion --tag $TAG --episodes 60 --warm_eps 10 \
     --T 20 --sparsemax --randn_prior --deadline 11 --arrival_dt 6 --m_div 8 --div_target $T --div_lr 0.02 \
     2>&1 | grep -E "^ep0(00|20|40)|^ep059" | grep -v warn
  echo "######## DIAG $TAG ########"
  python "$SCRATCH/diag_base_diffusion.py" $TAG seq_het_gauss 2>&1 | grep -E "DIFF_TAG|探针 E|w=0\.|gap" | grep -v -E "FutureWarning|warn"
done
echo "==== DONE sweep ===="
