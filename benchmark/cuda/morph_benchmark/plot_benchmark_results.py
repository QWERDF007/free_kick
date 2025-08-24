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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        
        # 获取性能数据
        time_mean = benchmark.get('cpu_time', 0) / 1000.0  # 转换为毫秒
        time_std = benchmark.get('cpu_time_std', 0) / 1000.0
        
        result = {
            'implementation': parsed['implementation'],
            'operation': parsed['operation'],
            'se_shape': parsed['se_shape'],
            'kernel_size': parsed['kernel_size'],
            'time_mean': time_mean,
            'time_std': time_std,
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
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
        
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
    # 按实现方式分组的平均性能
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: 不同实现方式的平均性能
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
    ax1.set_title('不同实现方式的平均性能')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, time in zip(bars1, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.2f}', ha='center', va='bottom')
    
    # 图2: 不同 kernel_size 的性能对比
    kernel_sizes = sorted(df['kernel_size'].unique())
    implementations = df['implementation'].unique()
    
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
        
        ax2.plot(kernel_sizes, avg_times_by_size, marker='o', linewidth=2, 
                markersize=6, label=impl)
    
    ax2.set_xlabel('Kernel Size')
    ax2.set_ylabel('平均执行时间 (ms)')
    ax2.set_title('不同 Kernel Size 的性能对比')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results, output_dir):
    """生成汇总报告"""
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
            print(f"     -> time: {result['time_mean']:.2f} ms")
        
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
    exit(main())
