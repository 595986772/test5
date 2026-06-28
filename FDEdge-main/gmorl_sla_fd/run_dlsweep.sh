#!/usr/bin/env bash
# deadline sweep: 同策略(dyn2_*14 ckpt)对一组 SLA deadline 评测可行性 -> SLA-可行前沿 vs deadline。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
OUT=result2/frac/dlsweep_dyn2.txt; : > "$OUT"
echo "deadline | diff(HV/feasHV/#feas) | gauss(HV/feasHV/#feas)" | tee -a "$OUT"
for D in 13 14 15 16 17 18 20; do
  L=result2/frac/_dlsweep_${D}.log
  python eval_frac.py --tags dyn2_diff14,dyn2_gauss14 --actors diffusion,mlp --k 10 --n_cand 1 --deadline_eval $D >"$L" 2>&1
  DLINE=$(grep "dyn2_diff14"  "$L" | tail -1 | awk '{printf "%s/%s/%s", $2,$3,$5}')
  GLINE=$(grep "dyn2_gauss14" "$L" | tail -1 | awk '{printf "%s/%s/%s", $2,$3,$5}')
  echo "   ${D}    | ${DLINE} | ${GLINE}" | tee -a "$OUT"
done
echo "=== DONE ===" | tee -a "$OUT"
