import os
import io
import json
import random
import asyncio
import multiprocessing
import threading

from pathlib import Path

import torch
import numpy as np
import traceback


class NetworkServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def run_server(self):
        """在新线程中运行服务器"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())
        loop.run_forever()

    async def start_server(self):
        """启动TCP服务器"""
        print(f'启动服务器')
        self.server = await asyncio.start_server(
            self.handle_client,
            self.host, 
            self.port
        )
        print(f'服务器启动成功')
        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader, writer):
        pass


def prepare_data(ok_dir, ng_dir, mask_dir, feat_dir, batch_size, patch_size):
        """准备数据"""
        ok_paths = list(ok_dir.iterdir())
        ok_paths = random.sample(ok_paths, k=min(100, len(ok_paths)))
        ng_paths = list(ng_dir.iterdir())
        ng_paths = random.sample(ng_paths, k=min(batch_size, len(ng_paths)))
        feat_paths = [feat_dir / f'{ok_dir.stem}.npy' for ok_dir in ok_paths]
        feats = []
        for feat_path in feat_paths:
            feats.append(np.load(str(feat_path)))
        feats = np.stack(feats, axis=0).mean(axis=0).reshape(1, patch_size, patch_size, -1) # HWC, e.g. [1, 32, 32, 1024]
        feats = feats.transpose(0, 3, 1, 2) # BHWC -> BCHW, e.g. [1, 1024, 32, 32]
        results = []
        feat_buffer = io.BytesIO()
        np.save(feat_buffer, feats)
        feat_bytes = feat_buffer.getvalue().hex()
        for ng_path in ng_paths:
            data = {
                'im_file': str(ng_path),
                'feats': feat_bytes,
            }
            results.append(data)
        return results


class NetworkDatasetProducer(NetworkServer):
    def __init__(self, host, port, total=32, num_workers=4):
        super().__init__(host, port)

        self.total = total
        self.num_workers = num_workers
        self.imgsz = 448
        self.patch_size = 448 // 14
        self.sample_num = 4
        self.batch_size = 8

        # 主线程准备数据的队列
        self.data_prepare_queue = multiprocessing.Queue(maxsize=4 * 10)
        # 子线程生产数据的队列
        self.data_queue = multiprocessing.Queue(maxsize=self.total * 10)

        self.used_subdirs = set()

    def get_data_dir(self, root_dir, classes):
        """获取数据目录"""
        root_dir = Path(root_dir)
        ok_dir = root_dir / 'traindata' / 'ok'
        ng_dir = root_dir / 'traindata' / 'ng'
        mask_dir = root_dir / 'traindata' / 'masks'
        feat_dir = root_dir / 'features'
        if classes:
            ok_sub_dirs = [sub_dir for sub_dir in ok_dir.iterdir() if sub_dir.is_dir() and sub_dir.name in classes]
        else:
            ok_sub_dirs = [sub_dir for sub_dir in ok_dir.iterdir() if sub_dir.is_dir()]
        ng_sub_dirs = [ng_dir / ok_dir.name for ok_dir in ok_sub_dirs]
        mask_sub_dirs = [mask_dir / ok_dir.name for ok_dir in ok_sub_dirs]
        feat_sub_dirs = [feat_dir / ok_dir.name for ok_dir in ok_sub_dirs]  
        orig_subdirs = [(ok_sub_dir, ng_sub_dir, mask_sub_dir, feat_sub_dir) for ok_sub_dir, ng_sub_dir, mask_sub_dir, feat_sub_dir in zip(ok_sub_dirs, ng_sub_dirs, mask_sub_dirs, feat_sub_dirs)]
        return orig_subdirs

    def sample_subdirs(self, orig_subdirs, k):
        # 计算还未使用的目录
        unused_subdirs = [d for d in orig_subdirs if d not in self.used_subdirs]
        
        # 如果未使用的目录数量为0,则重置已使用集合
        if len(unused_subdirs) == 0:
            self.used_subdirs.clear()
            unused_subdirs = orig_subdirs
        # 如果数量不足k,从已使用的目录中随机抽取补充
        elif len(unused_subdirs) < k:
            used_list = list(self.used_subdirs)
            additional = random.sample(used_list, k=k-len(unused_subdirs))
            unused_subdirs.extend(additional)
        
        # 从未使用的目录中随机抽取
        to_sample = min(k, len(unused_subdirs))
        subdirs = random.sample(unused_subdirs, k=to_sample)
        
        # 记录本次使用的目录
        self.used_subdirs.update(subdirs)
        print(f'使用目录: {[ok_dir.name for ok_dir, ng_dir, mask_dir, feat_dir in subdirs]}')

        return subdirs


    def start_data_producer(self, root_dir, classes):
        """启动数据生产者进程"""
        print(f'启动数据生产者进程')
        orig_subdirs = self.get_data_dir(root_dir, classes)
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            while True:
                subdirs = self.sample_subdirs(orig_subdirs, self.sample_num)
                tasks = []
                for ok_sub_dir, ng_sub_dir, mask_sub_dir, feat_sub_dir in subdirs:
                    task = pool.apply_async(prepare_data,
                                            args=(ok_sub_dir, ng_sub_dir, mask_sub_dir, feat_sub_dir, self.batch_size, self.patch_size))
                    tasks.append(task)
                
                # 等待所有任务完成
                for task in tasks:
                    results = task.get()
                    for data in results:
                        self.data_queue.put(data)

        

    def stop(self):
        for p in self.producer_processes:
            print(f'终止进程: {p.pid}')
            p.terminate()


    async def handle_client(self, reader, writer):
        """处理客户端连接

        Args:
            reader: StreamReader对象, 用于从客户端读取数据
            writer: StreamWriter对象, 用于向客户端发送数据
        """
        addr = writer.get_extra_info('peername')
        addr = f'{addr[0]}:{addr[1]}'
        print(f'客户端 {addr} 已连接')
        while True:
            try:
                # 等待客户端请求
                data = await asyncio.wait_for(reader.read(1024), timeout=None)  # 设置10秒超时, None 一直等待
                data_dict = json.loads(data.decode())
                if data_dict.get('method', None) != 'GET':
                    continue
                    
                # 等待队列中有数据
                batch = self.data_queue.get()

                # 将数据转换为JSON格式并发送
                data = json.dumps(batch).encode()
                header = bytes([0x95, 0x95]) + len(data).to_bytes(4, 'big')
                tail = bytes([0x95, 0x95])
                # print(f'发送数据: {len(data)}')
                writer.write(header + data + tail)
                await writer.drain()
            except Exception as e:
                # traceback.print_exc()
                # print(f'处理客户端 {addr} 请求时发生错误: {e}')
                break
            
        print(f'客户端 {addr} 断开连接')
        writer.close()
        await writer.wait_closed()
        

if __name__ == '__main__':
    data_dir = Path('/data2/wt/aquila_data')
    producer = NetworkDatasetProducer(host='127.0.0.1', port=11456, total=32, num_workers=4)
    producer.start_data_producer(data_dir, None)
    producer.stop()
    print('结束')
