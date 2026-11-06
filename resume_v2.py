#!/usr/bin/env python3
"""멈춘 v2 재학습과 2026년 평가를 이어 한다 (2026-11-02). 끝난 단계는 v2/run/<단계>.done으로 건너뛴다.

  python resume_v2.py           # 지금 상태만 보여 준다
  python resume_v2.py --start   # 앱과 분리해 run_final_all.py, run_holdout_photos.py를 띄운다(이미 돌고 있으면 띄우지 않음)
"""
import argparse
import glob
import os
import subprocess

VENV = r"C:\Users\Guest_user\venvs\rne\Scripts"
PY, PYW = os.path.join(VENV, "python.exe"), os.path.join(VENV, "pythonw.exe")
RUNNERS = {"run_final_all.py": "v2_runner.out", "run_holdout_photos.py": "v2_photos.out"}
ALL_STEPS = ["search", "search_wo", "base_lightgbm", "final_lightgbm", "photo_eval_lightgbm", "base_catboost", "final_catboost",
             "photo_eval_catboost", "whatif", "ablate", "photo_cmp", "sensitivity", "memo_effects", "holdout_base",
             "photo26_fetch", "photo26_retry", "photo26_embed", "holdout_photo", "holdout_photo_base"]


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")


def running():
    out = ps("[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }").stdout
    return [r for r in RUNNERS if r in out]


def wmi_start(code):
    """앱의 프로세스 묶음 밖에서 띄운다(Start-Process로 띄우면 앱 정리 때 같이 꺼진 적이 있다)."""
    line = f'"{PYW}" -X utf8 -c "{code}"'.replace("'", "''")
    cwd = os.getcwd().replace("'", "''")
    r = ps(f"Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{line}'; CurrentDirectory='{cwd}'}} "
           "| ForEach-Object { $_.ReturnValue }")
    if r.stdout.strip() != "0":
        raise SystemExit(f"실행 실패: {r.stdout} {r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    a = ap.parse_args()
    done = {os.path.basename(p)[:-5] for p in glob.glob("v2/run/*.done")}
    print("끝난 단계:", ", ".join(s for s in ALL_STEPS if s in done) or "없음")
    print("남은 단계:", ", ".join(s for s in ALL_STEPS if s not in done) or "없음")
    now = running()
    print("돌고 있는 실행기:", ", ".join(now) or "없음")
    if not a.start:
        return
    for script, out in RUNNERS.items():
        if script in now:
            continue
        wmi_start(f"import subprocess; subprocess.call([r'{PY}','-X','utf8','-u','{script}'], stdout=open('{out}','a'), "
                  "stderr=subprocess.STDOUT, creationflags=0x08000000)")
        print("시작:", script)
    print("진행은 v2/run/pipeline.log, v2/run/pipeline_photos.log에서 본다")


if __name__ == "__main__":
    main()
