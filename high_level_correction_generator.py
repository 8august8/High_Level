#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高层次纠正信息生成器
使用Gemini API基于图片序列和现有标注生成高层次的纠正信息
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime

# 导入配置
from config import (
    GEMINI_CONFIG,
    PROCESSING_CONFIG,
    PROMPT_CONFIG,
    PATH_CONFIG
)


class HighLevelAnalysis(BaseModel):
    """高层次分析结果的数据模型"""
    reasoning: str  # 推理过程（英文，简洁）
    avoidance_high_level: str  # 高层次避免措施（英文，简洁）
    correction_high_level: str  # 高层次纠正措施（英文，简洁）
    

class HighLevelCorrectionGenerator:
    """高层次纠正信息生成器类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            api_key: Gemini API密钥，如果不提供则从环境变量GEMINI_API_KEY读取
        """
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        
        self.client = genai.Client()
        self.model = GEMINI_CONFIG["model"]
        
        # 创建输出目录
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载图片"""
        full_path = Path('data') / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"图片文件不存在，已尝试路径: {full_path}")
        
        return Image.open(full_path)
    
    def _extract_episode_id(self, episode_data: Dict[str, Any]) -> int:
        """从episode数据中提取真实的episode ID"""
        # 从video路径提取episode ID
        video_path = episode_data.get('video', '')
        
        if video_path:
            # 使用正则表达式提取episode ID，格式：task_name/videos/episode_XX_cam_high.mp4
            match = re.search(r'episode_(\d+)_cam_high\.mp4', video_path)
            if match:
                return int(match.group(1))
        
        # 如果失败，返回0
        return 0
    
    def _load_all_episode_images(self, episode_data: Dict[str, Any]) -> List[Image.Image]:
        """加载episode中的所有图片，包括序列图片和关键帧"""
        images = []
        image_paths = []
        
        # 加载序列中的所有图片
        episode_images = episode_data.get('images', [])
        for img_path in episode_images:
            try:
                img = self._load_image(img_path)
                images.append(img)
                image_paths.append(img_path)
            except Exception as e:
                print(f"  警告: 无法加载图片 {img_path}: {e}")
        
        print(f"  成功加载 {len(images)} 张图片")
        
        if len(images) > 100:
            print(f"  图片数量 ({len(images)}) 较多，请注意API限制")
        
        return images
    
    def _create_analysis_prompt(self, episode_data: Dict[str, Any]) -> str:
        """创建用于生成高层次分析的提示词"""
        # 获取失败的子任务描述
        failed_subtask_description = 'Unknown'
        if episode_data['failure_subtask'].isdigit():
            subtask_idx = int(episode_data['failure_subtask']) - 1
            if 0 <= subtask_idx < len(episode_data['subtasks']):
                failed_subtask_description = episode_data['subtasks'][subtask_idx]
        
        # 使用配置中的提示词模板
        prompt = PROMPT_CONFIG["high_level_analysis_prompt"].format(
            task=episode_data['task'],
            failure_subtask=episode_data['failure_subtask'],
            failed_subtask_description=failed_subtask_description,
            failure_type=episode_data['failure_type'],
            low_level_avoidance=episode_data['avoidance'][0]['low_level'],
            low_level_correction=episode_data['correction'][0]['low_level']    
        )
        return prompt
    
    def _call_gemini_api(self, prompt: str, images: List[Image.Image], response_schema: type) -> Any:
        """调用Gemini API进行图片理解和文本生成"""
        try:
            # 准备内容列表
            contents = [images]
            contents.extend([prompt])

            config = types.GenerateContentConfig(
                response_mime_type=GEMINI_CONFIG["response_mime_type"],
                response_schema=response_schema,
                temperature=GEMINI_CONFIG["temperature"],  # 较低的温度以获得更一致的结果
            )
            # 调用API
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            return response.parsed
            
        except Exception as e:
            print(f"API调用失败: {e}")
            return None

    def _episode_id_count(self, task_name: str) -> int:
        """统计任务中episode的数量"""
        annotation_file = Path('data') / f"{task_name}/{task_name}_annotations.json"
        with open(annotation_file, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
        return len(episodes)
    
    def generate_high_level_analysis(self, episode_data: Dict[str, Any]) -> Optional[HighLevelAnalysis]:
        """生成单个episode的高层次分析"""
        print(f"正在处理高层次分析: {episode_data.get('video', 'Unknown')}")
        
        try:
            # 加载所有episode图片
            images = self._load_all_episode_images(episode_data)
            
            if not images:
                print(f"  错误: 未能加载任何图片")
                return None
            
            # 创建提示词
            prompt = self._create_analysis_prompt(episode_data)
            
            # 调用API
            print(f"  调用Gemini API生成高层次分析...")
            result = self._call_gemini_api(prompt, images, HighLevelAnalysis)
            
            if result:
                print(f"  ✅ 成功生成高层次分析")
                return result
            else:
                print(f"  ❌ API调用失败")
                return None
                
        except Exception as e:
            print(f"  ❌ 处理过程中出错: {e}")
            return None
    
    def process_single_episode(self, episode_data: Dict[str, Any], episode_id: int = None) -> Dict[str, Any]:
        """处理单个episode，生成高层次分析"""

        # 如果episode_id为None，则从数据中提取
        if episode_id is None:
            episode_id = self._extract_episode_id(episode_data)
        
        result = {
            "episode_id": f"episode_{episode_id}",
            "task": episode_data.get('task', ''),
            "failure_type": episode_data.get('failure_type', ''),
            "failure_subtask": episode_data.get('failure_subtask', ''),
        }
        
        if episode_data.get('failure_detection', '') == 'yes':
            # 生成高层次分析
            high_level_analysis = self.generate_high_level_analysis(episode_data)
            if high_level_analysis:
                result.update(high_level_analysis.model_dump())
        else:
            result.update({
                "avoidance_high_level": "",
                "correction_high_level": "",
                "reasoning": ""
            })
        
        # 添加延迟以避免API限制
        time.sleep(PROCESSING_CONFIG["api_call_delay"])
        
        return result
    
    def process_annotation_file(self, annotation_file: str) -> None:
        """处理标注文件，生成所有episode的高层次信息"""
        annotation_path = Path('data') / annotation_file
        
        if not annotation_path.exists():
            print(f"标注文件不存在: {annotation_path}")
            return
        
        print(f"正在处理标注文件: {annotation_file}")
        
        # 读取标注数据
        with open(annotation_path, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
        
        # 获取任务名称（从文件路径提取）
        task_name = annotation_path.stem.replace('_annotations', '')
        task_dir = self.output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        for i, episode_data in enumerate(episodes):
            print(f"\n处理 {i+1}/{len(episodes)} 个数据点")
            
            # 从数据中提取真实的episode ID
            episode_id = self._extract_episode_id(episode_data)
            print(f"Episode ID: episode_{episode_id}")
            
            result = self.process_single_episode(episode_data, episode_id)
            
            episode_output_file = task_dir / f"episode_{episode_id}_high_level.json"
            with open(episode_output_file, 'w', encoding='utf-8') as f:
                json_indent = 4
                json.dump(result, f, ensure_ascii=False, indent=json_indent)
                
            print(f"已保存: {episode_output_file}")
            
            # 添加延迟以避免API限制
            time.sleep(PROCESSING_CONFIG["episode_processing_delay"])
    
    def process_all_tasks(self) -> None:
        """处理data目录下的所有任务"""
        data_dir = Path('data')
        
        for task_dir in data_dir.iterdir():
            if task_dir.is_dir():
                annotation_file = task_dir / f"{task_dir.name}_annotations.json"
                if annotation_file.exists():
                    print(f"\n开始处理任务: {task_dir.name}")
                    self.process_annotation_file(str(annotation_file.relative_to(data_dir)))
                else:
                    print(f"跳过 {task_dir.name}: 未找到标注文件")


def main():
    """主函数"""
    print("高层次纠正信息生成器")
    print("=" * 50)
    
    # 检查API密钥
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 未设置GEMINI_API_KEY环境变量")
        print("请设置您的Gemini API密钥:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # 创建生成器
    generator = HighLevelCorrectionGenerator()
    
    # 询问用户处理选项
    print("\n请选择处理选项:")
    print("1. 处理所有任务")
    print("2. 处理单个任务")
    print("3. 处理单个episode")
    
    choice = input("请输入选择 (1 或 2 或 3): ").strip()
    
    if choice == "1":
        generator.process_all_tasks()
    elif choice == "2" or choice == "3":
        # 列出可用的任务
        data_dir = Path('data')
        available_tasks = []
        for task_dir in data_dir.iterdir():
            if task_dir.is_dir():
                annotation_file = task_dir / f"{task_dir.name}_annotations.json"
                if annotation_file.exists():
                    available_tasks.append(task_dir.name)
        
        if not available_tasks:
            print("未找到可处理的任务")
            return
        
        print("\n可用的任务:")
        for i, task in enumerate(available_tasks, 1):
            print(f"{i}. {task}")
        
        try:
            task_choice = int(input("请选择任务编号: ")) - 1
            if 0 <= task_choice < len(available_tasks):
                selected_task = available_tasks[task_choice]
                annotation_file = f"{selected_task}/{selected_task}_annotations.json"
                if choice == "2":
                    generator.process_annotation_file(annotation_file)
                elif choice == "3":
                    episode_id_count = generator._episode_id_count(available_tasks[task_choice])
                    print(f"\n可用的episode:\nepisode_0-episode_{episode_id_count-1}")
                    
                    try:
                        episode_choice = int(input("请选择episode编号: "))
                        annotation_path = Path('data') / annotation_file
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            episode_data = json.load(f)
                        if 0 <= episode_choice < episode_id_count:
                            result = generator.process_single_episode(episode_data[episode_choice], episode_choice)
                        
                            task_dir = generator.output_dir / selected_task
                            task_dir.mkdir(exist_ok=True)
                            
                            episode_output_file = task_dir / f"episode_{episode_choice}_high_level.json"
                            with open(episode_output_file, 'w', encoding='utf-8') as f:
                                json_indent = 4
                                json.dump(result, f, ensure_ascii=False, indent=json_indent)
                                
                            print(f"已保存: {episode_output_file}")
                            
                        else:
                            print("无效的episode选择")
                    except ValueError:
                        print("请输入有效的数字")
            else:
                print("无效的任务选择")
        except ValueError:
            print("请输入有效的数字")
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()
