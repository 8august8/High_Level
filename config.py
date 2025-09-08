#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 用于自定义高层次纠正信息生成器的行为
"""

# ===================== Gemini API配置 =====================
GEMINI_CONFIG = {
    "model": "gemini-2.5-pro",          # 使用的模型名称
    "temperature": 0.1,                 # 较低的温度以获得更一致的结果
    "response_mime_type": "application/json",  # API响应格式
    "max_retries": 2,                   # API调用失败时的重试次数
    "retry_delay": 3,                   # 重试之间的延迟（秒）
}

# ===================== 处理配置 =====================
PROCESSING_CONFIG = {
    "api_call_delay": 2,                # API调用之间的延迟（秒），避免API限制
    "episode_processing_delay": 2,       # episode处理之间的延迟（秒）
    "skip_existing_results": False,     # 是否跳过已存在的结果文件
}

# ===================== 提示词模板配置 =====================
PROMPT_CONFIG = {
    # 主要分析提示词模板（来自 _create_analysis_prompt 方法）
    "high_level_analysis_prompt": """
# Role:
You are an expert in robotics and task planning analysis. Your goal is to identify the root cause of a robot's manipulation failure and provide high-level, strategic advice for avoidance and correction.

# Context:
You will be provided with a sequence of images from a robotic task, two keyframes ('Avoid' and 'Correct'), and metadata about the task. The 'Avoid' frame is captured just before the failure occurs. The 'Correct' frame shows a state that should apply a correction. The two keyframes also has existing low-level annotations (like arrows for movement/rotation or gripper state changes). Your task is to go beyond these low-level details.

# Input:

- Task: {task}
- Failed Subtask: Step {failure_subtask} - {failed_subtask_description}
- Failure Type: {failure_type}
- Low-level Correction: {low_level_correction}
- Low-level Avoidance: {low_level_avoidance}

# Instructions:

Based on all the provided information, perform the following three tasks. Your response MUST be a single JSON object with three keys: `reasoning`, `avoidance_high_level`, `correction_high_level`.

1. `reasoning`: Provide a concise, high-level analysis of why the failure happened. Focus on strategic errors, not just physical misalignments. For example, 'The robot failed to account for the object's center of mass' or 'The task plan incorrectly sequenced the left and right arms'. This will be used for Chain-of-Thought fine-tuning.

2. `avoidance_high_level`: Based on the 'Avoid' keyframe and the overall context, suggest a high-level strategic change to prevent this failure in the future. Do NOT use low-level commands like 'move left', 'go up', or 'rotate clockwise'. Instead, use conceptual language.
- Good Example: 'Adjust the gripper's approach angle to align with the object's primary axis.'
- Bad Example: 'Move the gripper forward and then down.'

3. `correction_high_level`: Based on the 'Correct' keyframe, describe a high-level strategy to recover from the failure state. Again, avoid low-level commands.
- Good Example: 'Release the object, reset the arm to a safe position, and re-attempt the grasp from a more stable orientation.'
- Bad Example: 'Open the gripper, move back, and try again.'"

Keep responses concise, practical, and focused on actionable high-level guidance rather than technical details.

# Output format:
"Ensure your final output is only the JSON object, without any additional text or explanations before or after it."
"""
}

# ===================== 路径配置 =====================
PATH_CONFIG = {
    "data_root": "data",                  # 数据根目录
    "frames_subdir": "frames",            # 帧目录名称
    "videos_subdir": "videos",            # 视频目录名称
}

# ===================== 批处理配置 =====================
BATCH_PROCESSING_CONFIG = {
    "enable_batch_mode": True,            # 是否启用批处理模式
    "batch_size": 5,                     # 批处理大小（同时处理的episode数）
    "save_intermediate_results": True,   # 是否保存中间结果
    "resume_from_checkpoint": True,      # 是否从检查点恢复
    "checkpoint_interval": 10,           # 检查点保存间隔（处理多少个episode后保存）
}