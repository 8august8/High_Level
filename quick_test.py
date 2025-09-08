#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 处理单个episode进行测试
"""

import os
import json
from pathlib import Path
from high_level_correction_generator import HighLevelCorrectionGenerator


def test_single_episode():
    """测试处理单个episode"""
    print("开始快速测试...")
    
    # 检查API密钥
    if not os.getenv('GEMINI_API_KEY'):
        print("错误: 请设置GEMINI_API_KEY环境变量")
        return
    
    # 选择第一个可用的任务进行测试
    data_dir = Path("data")
    
    for task_dir in data_dir.iterdir():
        if task_dir.is_dir():
            annotation_file = task_dir / f"{task_dir.name}_annotations.json"
            if annotation_file.exists():
                print(f"找到任务: {task_dir.name}")
                
                # 读取第一个episode进行测试
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    episodes = json.load(f)
                
                if episodes:
                    print(f"测试第一个episode...")
                    
                    # 创建生成器
                    generator = HighLevelCorrectionGenerator()
                    
                    # 处理第一个episode
                    episode_data = episodes[0]
                    
                    result = generator.process_single_episode(episode_data)
                    
                    # 保存测试结果
                    test_output_file = Path("output") / f"test_result.json"
                    with open(test_output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    print(f"测试完成！结果已保存到: {test_output_file}")
                    return
                
                break
    
    print("未找到可测试的数据")


if __name__ == "__main__":
    test_single_episode()
