#!/usr/bin/env python3
"""
Benchmark 结果可视化脚本
解析 Google Benchmark 的 JSON 输出并绘制性能曲线
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# 设置中文字体支持和字符编码
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 10

# 设置后端和编码
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def setup_matplotlib_fonts():
    """设置 matplotlib 字体配置，避免字符显示警告"""
    import matplotlib.font_manager as fm
    
    # 检查可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 优先级顺序的字体列表
    preferred_fonts = [
        'Microsoft YaHei',    # Windows 默认中文字体
        'SimHei',            # 黑体
        'STSong',            # 华文宋体
        'Arial Unicode MS',   # macOS 中文支持
        'Noto Sans CJK',     # Linux 中文支持
        'WenQuanYi Micro Hei', # Linux 文泉驿
        'DejaVu Sans',       # 通用字体
        'Arial',             # 基础字体
        'sans-serif'         # 系统默认
    ]
    
    # 找到第一个可用的字体
    selected_font = 'sans-serif'  # 默认
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 更新字体配置
    matplotlib.rcParams['font.sans-serif'] = [selected_font] + preferred_fonts
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 额外的配置来避免警告
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    
    print(f"使用字体: {selected_font}")

def setup_kernel_size_ticks(ax, kernel_sizes):
    """为 kernel size 轴设置更好的刻度显示"""
    if not kernel_sizes:
        return
    
    # 设置所有 kernel_size 作为刻度（线性刻度）
    ax.set_xticks(kernel_sizes)
    ax.set_xticklabels([str(k) for k in kernel_sizes])
    ax.tick_params(axis='x', rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)

def parse_benchmark_name(name):
    """
    从 benchmark 名称中解析参数
    例如: "CUDA_V0_Erode_ELLIPSE_5/0/5/2/real_time"
    解析出: implementation="CUDA_V0", operation="Erode", se_shape="ELLIPSE", kernel_size=5
    """
    # 移除末尾的 "/real_time" 或 "/cpu_time" 等后缀
    if '/' in name:
        name = name.split('/')[0]
    
    # 按下划线分割名称
    parts = name.split('_')
    
    if len(parts) >= 4:
        # 前两部分是实现方式 (如 "CUDA_V0", "OpenCV")
        if parts[0] == "CUDA" and len(parts) >= 2:
            implementation = parts[0] + "_" + parts[1]  # "CUDA_V0"
            operation = parts[2]                        # "Erode"
            se_shape = parts[3]                         # "ELLIPSE"
            kernel_size = int(parts[4])                 # 5
        elif parts[0] == "OpenCV":
            implementation = parts[0]                    # "OpenCV"
            operation = parts[1]                        # "Erode"
            se_shape = parts[2]                         # "ELLIPSE"
            kernel_size = int(parts[3])                 # 5
        else:
            return None
        
        return {
            'implementation': implementation,
            'operation': operation,
            'se_shape': se_shape,
            'kernel_size': kernel_size
        }
    
    return None

def parse_benchmark_json(json_file_path):
    """解析 benchmark JSON 文件"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for benchmark in data.get('benchmarks', []):
        name = benchmark['name']
        
        # 从名称中解析参数
        parsed = parse_benchmark_name(name)
        if not parsed:
            continue
        
        # 获取性能数据和时间单位
        time_mean = benchmark.get('cpu_time', 0)
        time_std = benchmark.get('cpu_time_std', 0)
        time_unit = benchmark.get('time_unit', 'us')  # 默认微秒
        
        # 根据时间单位转换为毫秒
        if time_unit == 'ns':  # 纳秒
            time_mean /= 1000000.0
            time_std /= 1000000.0
        elif time_unit == 'us':  # 微秒
            time_mean /= 1000.0
            time_std /= 1000.0
        elif time_unit == 'ms':  # 毫秒
            pass  # 已经是毫秒，不需要转换
        elif time_unit == 's':  # 秒
            time_mean *= 1000.0
            time_std *= 1000.0
        
        result = {
            'implementation': parsed['implementation'],
            'operation': parsed['operation'],
            'se_shape': parsed['se_shape'],
            'kernel_size': parsed['kernel_size'],
            'time_mean': time_mean,
            'time_std': time_std,
            'time_unit': time_unit,  # 保留原始时间单位
            'name': name  # 保留原始名称用于调试
        }
        
        results.append(result)
    
    return results

def plot_performance_curves(results, output_dir="plots"):
    """绘制性能曲线"""
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 转换为 DataFrame 便于分析
    df = pd.DataFrame(results)
    
    # 按操作类型分组绘制
    operations = df['operation'].unique()
    
    for operation in operations:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{operation} 操作性能对比', fontsize=16, fontweight='bold')
        
        # 按结构元素形状分组
        se_shapes = df[df['operation'] == operation]['se_shape'].unique()
        
        for i, se_shape in enumerate(se_shapes):
            if i >= 4:  # 最多4个子图
                break
                
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 筛选数据
            mask = (df['operation'] == operation) & (df['se_shape'] == se_shape)
            data = df[mask]
            
            # 按实现方式分组
            implementations = data['implementation'].unique()
            
            for impl in implementations:
                impl_data = data[data['implementation'] == impl]
                if len(impl_data) > 0:
                    # 按 kernel_size 排序
                    impl_data = impl_data.sort_values('kernel_size')
                    
                    # 绘制曲线
                    ax.errorbar(impl_data['kernel_size'], impl_data['time_mean'], 
                               yerr=impl_data['time_std'], 
                               marker='o', linewidth=2, markersize=6,
                               label=impl, capsize=4)
            
            ax.set_xlabel('Kernel Size')
            ax.set_ylabel('执行时间 (ms)')
            ax.set_title(f'{se_shape} 形状')
            ax.legend()
            # 只保留 y 轴的对数刻度，x 轴使用线性刻度
            ax.set_yscale('log')
            
            # 设置更好的 kernel size 刻度显示
            kernel_sizes = sorted(data['kernel_size'].unique()) if len(data) > 0 else []
            setup_kernel_size_ticks(ax, kernel_sizes)
        
        # 隐藏多余的子图
        for i in range(len(se_shapes), 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{operation}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制综合对比图
    plot_comprehensive_comparison(df, output_dir)

def plot_comprehensive_comparison(df, output_dir):
    """绘制综合对比图"""
    # 创建更详细的对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('综合性能对比分析', fontsize=16, fontweight='bold')
    
    # 图1: 不同实现方式的总体平均性能
    implementations = df['implementation'].unique()
    avg_times = []
    impl_names = []
    
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        avg_time = impl_data['time_mean'].mean()
        avg_times.append(avg_time)
        impl_names.append(impl)
    
    bars1 = ax1.bar(impl_names, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('平均执行时间 (ms)')
    ax1.set_title('总体平均性能\n(所有操作、形状、核大小的平均)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, time in zip(bars1, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.2f}', ha='center', va='bottom')
    
    # 图2: 按操作类型分组的平均性能
    operations = df['operation'].unique()
    op_data = []
    for op in operations:
        op_avg = []
        for impl in implementations:
            impl_op_data = df[(df['implementation'] == impl) & (df['operation'] == op)]
            if len(impl_op_data) > 0:
                op_avg.append(impl_op_data['time_mean'].mean())
            else:
                op_avg.append(0)
        op_data.append(op_avg)
    
    x = np.arange(len(implementations))
    width = 0.12
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (op, data) in enumerate(zip(operations, op_data)):
        ax2.bar(x + i * width, data, width, label=op, color=colors[i % len(colors)])
    
    ax2.set_xlabel('实现方式')
    ax2.set_ylabel('平均执行时间 (ms)')
    ax2.set_title('按操作类型分组的平均性能')
    ax2.set_xticks(x + width * (len(operations) - 1) / 2)
    ax2.set_xticklabels(implementations, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 按核大小分组的性能趋势
    kernel_sizes = sorted(df['kernel_size'].unique())
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        avg_times_by_size = []
        
        for size in kernel_sizes:
            size_data = impl_data[impl_data['kernel_size'] == size]
            if len(size_data) > 0:
                avg_time = size_data['time_mean'].mean()
                avg_times_by_size.append(avg_time)
            else:
                avg_times_by_size.append(np.nan)
        
        ax3.plot(kernel_sizes, avg_times_by_size, marker='o', linewidth=2, 
                markersize=6, label=impl)
    
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('平均执行时间 (ms)')
    ax3.set_title('不同核大小的性能趋势\n(所有操作和形状的平均)')
    ax3.legend()
    # 只保留 y 轴的对数刻度，x 轴使用线性刻度
    ax3.set_yscale('log')
    
    # 设置更好的 kernel size 刻度显示
    setup_kernel_size_ticks(ax3, kernel_sizes)
    
    # 图4: 按形状分组的平均性能
    se_shapes = df['se_shape'].unique()
    shape_data = []
    for shape in se_shapes:
        shape_avg = []
        for impl in implementations:
            impl_shape_data = df[(df['implementation'] == impl) & (df['se_shape'] == shape)]
            if len(impl_shape_data) > 0:
                shape_avg.append(impl_shape_data['time_mean'].mean())
            else:
                shape_avg.append(0)
        shape_data.append(shape_avg)
    
    x = np.arange(len(implementations))
    width = 0.25
    shape_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (shape, data) in enumerate(zip(se_shapes, shape_data)):
        ax4.bar(x + i * width, data, width, label=shape, color=shape_colors[i % len(shape_colors)])
    
    ax4.set_xlabel('实现方式')
    ax4.set_ylabel('平均执行时间 (ms)')
    ax4.set_title('按结构元素形状分组的平均性能')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(implementations, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results, output_dir):
    """生成汇总报告"""
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # 按实现方式分组的统计
    summary = df.groupby('implementation').agg({
        'time_mean': ['mean', 'std', 'min', 'max'],
        'kernel_size': 'count'
    }).round(4)
    
    # 保存汇总报告
    summary.to_csv(f'{output_dir}/summary_report.csv')
    
    # 生成性能排名
    avg_performance = df.groupby('implementation')['time_mean'].mean().sort_values()
    performance_ranking = pd.DataFrame({
        'Implementation': avg_performance.index,
        'Average Time (ms)': avg_performance.values,
        'Rank': range(1, len(avg_performance) + 1)
    })
    
    performance_ranking.to_csv(f'{output_dir}/performance_ranking.csv', index=False)
    
    print("汇总报告已生成:")
    print(f"- 详细统计: {output_dir}/summary_report.csv")
    print(f"- 性能排名: {output_dir}/performance_ranking.csv")
    
    return summary, performance_ranking

def main():
    parser = argparse.ArgumentParser(description='Benchmark 结果可视化工具')
    parser.add_argument('json_file', help='Benchmark JSON 结果文件路径')
    parser.add_argument('-o', '--output', default='plots', help='输出目录 (默认: plots)')
    parser.add_argument('--no-plots', action='store_true', help='不生成图表，只生成报告')
    
    args = parser.parse_args()
    
    # 设置字体配置，避免警告
    setup_matplotlib_fonts()
    
    # 检查输入文件
    if not Path(args.json_file).exists():
        print(f"错误: 找不到文件 {args.json_file}")
        return 1
    
    print(f"正在解析 benchmark 结果文件: {args.json_file}")
    
    try:
        # 解析 JSON 文件
        results = parse_benchmark_json(args.json_file)
        
        if not results:
            print("错误: 没有找到有效的 benchmark 结果")
            return 1
        
        print(f"成功解析 {len(results)} 条 benchmark 结果")
        
        # 打印解析结果示例
        print("\n解析结果示例:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result['name']}")
            print(f"     -> implementation: {result['implementation']}")
            print(f"     -> operation: {result['operation']}")
            print(f"     -> se_shape: {result['se_shape']}")
            print(f"     -> kernel_size: {result['kernel_size']}")
            print(f"     -> time: {result['time_mean']:.2f} ms (原始单位: {result['time_unit']})")
        
        # 生成汇总报告
        summary, ranking = generate_summary_report(results, args.output)
        
        # 打印性能排名
        print("\n性能排名:")
        print(ranking.to_string(index=False))
        
        if not args.no_plots:
            # 生成图表
            print(f"\n正在生成图表到目录: {args.output}")
            plot_performance_curves(results, args.output)
            print("图表生成完成!")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
