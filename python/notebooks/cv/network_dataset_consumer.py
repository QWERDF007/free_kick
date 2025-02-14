import os
import io
import time
import json
import asyncio
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset


class NetworkDatasetConsumer(Dataset):
    def __init__(self, host, port, total):
        self.host = host
        self.port = port
        self.total = total
        self.reader = None
        self.writer = None

    def __len__(self):
        return self.total
    
    def connect(self):
        # 创建事件循环并建立连接
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._connect())

    async def _connect(self):
        """建立TCP连接"""
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print(f'连接到服务器 {self.host}:{self.port}')

    def ensure_connection(self):
        """确保连接已建立"""
        if self.reader is None or self.writer is None:
            self.connect()

    def get_data(self):
        """获取数据集中的一个样本"""
        loop = asyncio.get_event_loop()
        batch = loop.run_until_complete(self._get_data())
        return batch

    async def _get_data(self):
        """获取数据集中的一个样本"""
        requeset = {'method': 'GET'}
        self.writer.write(json.dumps(requeset).encode())
        await self.writer.drain()

        # 读取4字节表头
        while True:
            header = await asyncio.wait_for(self.reader.read(2), timeout=None)
            if header == bytes([0x95, 0x95]):
                break
            
        # 读取4字节数据长度
        length_bytes = await asyncio.wait_for(self.reader.read(4), timeout=None)
        data_length = int.from_bytes(length_bytes, 'big')
        # print(f'读取数据长度: {data_length}')
        # 根据长度读取完整数据
        data = b''
        remaining = data_length
        while remaining > 0:
            chunk = await asyncio.wait_for(self.reader.read(min(1024, remaining)), timeout=None)
            data += chunk
            remaining -= len(chunk)
        tail = await asyncio.wait_for(self.reader.read(2), timeout=None)
        if tail != bytes([0x95, 0x95]):
            print(f'无效的表尾: {tail}')
        batch = json.loads(data.decode())
        feat_bytes = bytes.fromhex(batch['feats'])
        feat_buffer = io.BytesIO(feat_bytes)
        feats = np.load(feat_buffer)
        batch['feats'] = feats
        return batch

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        self.ensure_connection()
        batch = self.get_data()
        # print(f'得到数据, pid: {os.getpid()}, idx: {idx}, batch: {batch}')
        return batch
    

if __name__ == "__main__":
    dataset = NetworkDatasetConsumer(host='127.0.0.1', port=11456, total=32)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    while True:
        t0 = time.time()
        for data in loader:
            t1 = time.time()
            print(f'耗时: {t1 - t0}')
            t0 = t1
            image_paths = data['im_file']
            diff_dir = set()
            for path in image_paths:
                p = Path(path)
                diff_dir.add(p.parent.name)
            print(len(image_paths), len(diff_dir), diff_dir)
            t0 = time.time()
