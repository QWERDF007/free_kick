# Morphology Benchmark 工具

这个工具用于测试不同 CUDA 实现和 OpenCV 的形态学操作性能，并生成可视化图表。

## 使用方法

### 1. 运行 Benchmark

#### 编译项目
```bash
cd /path/to/free_kick
cmake --build build --target free_kick_morph_benchmark
```

#### 运行 Benchmark 并输出 JSON 格式
```bash
# 运行所有测试并输出到 JSON 文件
./build/bin/Release/free_kick_morph_benchmark --benchmark_format=json --benchmark_out=benchmark_results.json

# 或者运行特定测试
./build/bin/Release/free_kick_morph_benchmark --benchmark_filter="CUDA.*" --benchmark_format=json --benchmark_out=cuda_results.json
```

#### 其他有用的参数
```bash
# 设置重复次数
--benchmark_repetitions=10

# 设置时间单位
--benchmark_time_unit=us

# 只运行一次（快速测试）
--benchmark_min_time=0s
```

### 2. 生成可视化图表

#### 运行可视化脚本
```bash
# 生成所有图表
python plot_benchmark_results.py benchmark_results.json

# 指定输出目录
python plot_benchmark_results.py benchmark_results.json -o my_plots

# 只生成报告，不生成图表
python plot_benchmark_results.py benchmark_results.json --no-plots
```

## 输出内容

### 1. Benchmark 结果
- **操作类型**: 6种 (腐蚀、膨胀、开运算、闭运算、顶帽、黑帽)
- **结构元素形状**: 3种 (矩形、椭圆形、十字形)
- **核大小**: 8种 (3, 5, 7, 9, 11, 15, 21, 31)
- **实现方式**: 4种 (OpenCV, CUDA V0, CUDA V1, CUDA V2)
- **总组合数**: 6 × 3 × 8 × 4 = 576 种组合

### 2. 生成的图表
- **按操作类型分组**: 每个操作生成一个包含4个子图的图表
- **按结构元素形状**: 每个形状显示不同实现方式的性能曲线
- **综合对比图**: 包含平均性能柱状图和 kernel size 影响曲线
- **性能排名**: 不同实现方式的性能排序

### 3. 生成的报告
- **summary_report.csv**: 详细的统计信息
- **performance_ranking.csv**: 性能排名表

## 图表说明

### 性能曲线图
- X轴: Kernel Size (对数刻度)
- Y轴: 执行时间 (对数刻度，毫秒)
- 不同颜色的线: 不同实现方式
- 误差条: 标准差范围

### 综合对比图
- 左图: 不同实现方式的平均性能柱状图
- 右图: 不同 kernel size 对性能的影响曲线

## 自定义配置

### 修改测试参数
在 `main.cpp` 中修改以下数组：
```cpp
// 操作类型
const std::vector<int> morph_ops = {...};

// 结构元素形状
const std::vector<int> se_shapes = {...};

// 核大小
const std::vector<int> kernel_sizes = {...};
```

### 修改 Python 脚本
- 调整图表大小: 修改 `figsize` 参数
- 更改颜色方案: 修改 `color` 数组
- 添加新的图表类型: 在 `plot_performance_curves` 函数中添加

## 故障排除

### 常见问题

1. **Python 包安装失败**
   ```bash
   # 使用 conda 安装
   conda install matplotlib numpy pandas
   
   # 或者升级 pip
   python -m pip install --upgrade pip
   ```

2. **中文字体显示问题**
   - Windows: 确保安装了 SimHei 字体
   - Linux: 安装中文字体包
   - macOS: 使用系统默认中文字体

3. **Benchmark 运行缓慢**
   - 减少 `kernel_sizes` 数组中的值
   - 使用 `--benchmark_min_time=0s` 参数
   - 减少重复次数

4. **内存不足**
   - 减少测试图片尺寸
   - 分批运行不同的操作类型

### 性能优化建议

1. **CUDA 版本选择**
   - 小核 (< 7): V0 或 V1 可能更快
   - 大核 (≥ 7): V2 通常性能更好
   - 复杂形状: V2 的优势更明显

2. **结构元素形状影响**
   - 矩形: 计算最简单，性能最好
   - 椭圆形: 中等复杂度
   - 十字形: 最复杂，性能差异最大

3. **核大小影响**
   - 小核: 内存带宽限制
   - 大核: 计算复杂度限制
   - 最佳性能通常在 7-15 之间

## 扩展功能

### 添加新的实现方式
1. 在 `morphology_unified.cuh` 中定义新的执行器
2. 在 `main.cpp` 中添加新的 benchmark 注册
3. 在 Python 脚本中添加相应的颜色和标签

### 添加新的性能指标
1. 在 C++ 代码中添加新的计数器
2. 在 Python 脚本中解析新的计数器
3. 在图表中显示新的指标

### 自定义图表样式
1. 修改 `matplotlib` 的样式设置
2. 添加自定义的颜色方案
3. 调整图表的布局和标签
