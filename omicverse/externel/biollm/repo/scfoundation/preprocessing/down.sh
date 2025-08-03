#!/usr/bin/env bash

project=$1
log_path="./log/${project}/"
[[ -d $log_path ]] || mkdir -p $log_path

_LOG_INFO() {
  echo -e "$(date "+[%Y-%m-%d %H:%M:%S]")\tINFO\t$2\t$1" >>./log/${project}/info.log
}
_LOG_ERROR() {
  echo -e "$(date "+[%Y-%m-%d %H:%M:%S]")\tERROR\t$2\t$1" >>./log/${project}/error.log
  exit $2
}
line=$*
if [[ $# -ne 3 ]];then
       _LOG_ERROR "$line" 2
fi

_LOG_INFO "$line" 0

file_name=$2
file_url=$3


file=$(basename $file_name)
dir=$(dirname $file_name)
[[ -d $dir ]] || mkdir -p $dir

curl_times=3

while [ $curl_times -gt 0 ]; do
  curl -C - -L -o $file_name $file_url
  [[ $? -eq 0 ]] && break
  if [[ $curl_times == 1 ]];then
    rm $file_name
    _LOG_ERROR "$line" 4
  fi
  curl_times=$((curl_times - 1))
  sleep 1
done

_LOG_INFO "$line" 200