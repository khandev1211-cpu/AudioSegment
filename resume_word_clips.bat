@echo off
rem Resume/continue the full-Quran word-level clip build (GPU)
rem Safe to run any time - it skips already-processed verses (checkpoint in dataset\word_clips\progress.json)
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
python -u scripts\build_word_clips.py --device cuda
pause