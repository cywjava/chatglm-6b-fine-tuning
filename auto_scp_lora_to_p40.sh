#!/bin/bash

pt_path="/home/train/check_points/"
new_pt_path=`ls -t "${pt_path}"|head -n 1`
scp_file_dir=${pt_path}${new_pt_path}

c_time=`stat "${scp_file_dir}"|grep -i "最近改动："|awk -F "最近改动：" '{print $2}'`
c_time=${c_time:0:19}
c_time_stamp=$(date -d "${c_time}" +%s)

currentTime=`date "+%Y-%m-%d %H:%M:%S"`
currentTimeStamp=$(date -d "$currentTime" +%s)

echo "最新文件：$scp_file_dir,创建时间：${c_time}"

let time_2=currentTimeStamp-c_time_stamp


function scp_p40 {
scp -P12211 "${scp_file_dir}"/chatglm-6b-lora.pt thudm@192.168.20.8:/home/thudm/check_points/
ssh -p12211 thudm@192.168.20.8 << remotessh
cd /home/thudm/generate_server
bash start_chat_server.sh
echo "启动成功!"
remotessh
echo "${scp_file_dir}" >> /home/train/scp.log
}



if [ $time_2  -ge 300 ]; then
	scped=`grep -i "${scp_file_dir}" /home/train/scp.log`
	if [ "" == "${scped}" ]; then
		scp_p40
	fi	
fi
