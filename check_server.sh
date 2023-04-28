#!/bin/bash
cd /home/thudm/generate_server
pid=`ps -ef|grep "chat_server.py"|grep -v "grep"|awk '{print $2}'`

s_time=`date "+%Y-%m-%d %H:%M:%S"`

if [ "${pid}" == "" ];then
	echo "${s_time} ===>: 进程不存在，准备启动" >> ./logs/check.log 
	bash /home/thudm/generate_server/start_chat_server.sh
else
        echo "${s_time} ===>: chat_server.py 已启动" >> ./logs/check.log
fi

