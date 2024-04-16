@REM 进入文件所在目录
cd /d %~dp0

@REM 打印上条命令是否成功
echo %errorlevel%

@REM 激活conda环境
call conda activate AD

@REM 打印上条命令是否成功
echo %errorlevel%

@REM 执行自动签到脚本
python auto_checkin.py