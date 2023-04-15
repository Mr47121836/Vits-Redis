import json
import os
import time
import redis as redis
import vitsService as service
import langid

# redis cache client

RedisCache = redis.StrictRedis(host="localhost", port=6379, db=0)

# the queue of expect to detect
DATA_INFO_QUEUE = "dataInfoQueue"

# slice size every foreach

BATCH_SIZE = 32
# server sleep when queue>0

SERVER_SLEEP = 0.1
# server sleep when queue=0
SERVER_SLEEP_IDLE = 0.5


def wav_generation_process():

    while True:
        # 从redis中获取预测图像队列
        queue = RedisCache.lrange(DATA_INFO_QUEUE, 0, BATCH_SIZE - 1)

        if len(queue) < 1:
            time.sleep(SERVER_SLEEP)
            continue

        print("wav_generation_process is running")

        # 遍历队列
        for realItem in queue:
            # step 1. 获取队列中的图像信息
            item = json.loads(realItem);

            dataInfo_Id = item.get("dataInfoId")
            speaker_Id = item.get("speakerId")
            speaker_text = item.get("speakerText")
            noise_Scale = float(item.get("noiseScale"))
            noise_ScaleW = float(item.get("noiseScaleW"))
            length_Scale = float(item.get("lengthScale"))

            wav_path = service.gen_wav(int(speaker_Id),speaker_text,dataInfo_Id,noise_Scale,noise_ScaleW,length_Scale)

            if wav_path is not None: # 把结果存入redis
                print("正在存入Redis...")
                RedisCache.hset(name=dataInfo_Id, key="consultOut", value=wav_path)
            else:
                print("音频路径为空...")
                RedisCache.hset(name=dataInfo_Id, key="consultOut", value="")

        RedisCache.ltrim(DATA_INFO_QUEUE, BATCH_SIZE, -1)
        time.sleep(SERVER_SLEEP)

if __name__ == '__main__':

    print("Start wav_generation_process...")
    print(os.getcwd())
    wav_generation_process()
