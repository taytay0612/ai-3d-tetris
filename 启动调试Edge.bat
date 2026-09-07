@echo off
rem 启动带调试端口的 Edge，用于 AI 控制 3D 俄罗斯方块
setlocal
set PROFILE=%LOCALAPPDATA%\ai_tetris_edge
if not exist "%PROFILE%" mkdir "%PROFILE%"
start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9222 --user-data-dir="%PROFILE%" --new-window --window-size=1400,1000 --no-first-run --no-default-browser-check "file:///E:/deepseek/test/3d-tetris.html"
echo Edge is starting with remote debugging on port 9222.
echo Please leave this window open while the AI plays.
timeout /t 5
