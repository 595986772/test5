#!/usr/bin/env bash
# 最终头条: 新主方法 diffusion m_div=1 (dyn2_diffM1) vs gauss, deadline sweep。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
OUT=result2/frac/dlsweep_diffM1.txt; : > "$OUT"
echo "deadline | diffM1(HV/feasHV/#feas) | gauss(HV/feasHV/#feas)" | tee -a "$OUT"
for D in 13 14 15 16 17 18 20; do
  L=result2/frac/_dlsweepM1_${D}.log
  python eval_frac.py --tags dyn2_diffM1,dyn2_gauss14 --actors diffusion,mlp --k 10 --n_cand 1 --deadline_eval $D >"$L" 2>&1
  DLINE=$(grep "dyn2_diffM1"  "$L" | tail -1 | awk '{printf "%s/%s/%s", $2,$3,$5}')
  GLINE=$(grep "dyn2_gauss14" "$L" | tail -1 | awk '{printf "%s/%s/%s", $2,$3,$5}')
  echo "   ${D}    | ${DLINE} | ${GLINE}" | tee -a "$OUT"
done
echo "=== DONE ===" | tee -a "$OUT"
