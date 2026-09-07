@echo off
rem Neural network score model plays in Edge
cd /d E:\deepseek\test
set PYTHONPATH=E:\deepseek\test
python -u -m ai_tetris.score_play_browser --model E:\deepseek\test\ai_tetris\score_net.pt --port 9222 --episodes 3 > E:\deepseek\test\ai_tetris\score_play_log.txt 2>&1
echo score play finished
pause
