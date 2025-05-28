import streamlit as st
import cv2
import numpy as np
import moviepy.editor as mp
from PIL import Image
import torch
import torch.nn as nn
from datetime import datetime

# 模拟AI预测模型
class SimplePredictor:
    def __init__(self):
        # 模拟训练好的模型参数
        self.weights = {
            'duration': -0.2,
            'brightness': 0.3,
            'motion': 0.4,
            'audio_volume': 0.1
        }
    
    def predict(self, features):
        score = sum(features[k]*self.weights[k] for k in self.weights)
        return min(max(score*100, 0), 100)  # 确保分数在0-100之间

# 视频分析函数
def analyze_video(video_path):
    # 基础分析
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # 亮度分析
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # 运动检测
    ret, next_frame = cap.read()
    diff = cv2.absdiff(frame, next_frame)
    motion = np.mean(diff)
    
    # 音频分析
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio
    volume = np.mean(np.abs(audio.to_soundarray()))
    
    return {
        "duration": duration,
        "brightness": brightness,
        "motion": motion,
        "audio_volume": volume
    }

# 生成优化建议
def get_suggestions(result):
    suggestions = []
    if result['duration'] > 60:
        suggestions.append("⚠️ 视频过长（建议30秒内）")
    if result['brightness'] < 100:
        suggestions.append("🌟 提高画面亮度")
    if result['motion'] < 20:
        suggestions.append("🎬 增加镜头运动")
    return suggestions

# 界面布局
st.set_page_config(page_title="短视频爆款检测器", layout="wide")
st.title("🎬 一键检测视频爆款概率")

# 上传视频
uploaded_file = st.file_uploader("上传你的短视频（MP4格式）", type=["mp4"])

if uploaded_file:
    # 保存临时文件
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 显示视频
    st.video("temp_video.mp4")
    
    # 分析视频
    with st.spinner('正在分析视频...'):
        analysis = analyze_video("temp_video.mp4")
    
    # 显示分析结果
    st.subheader("📊 视频体检报告")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("时长（秒）", round(analysis['duration'], 1))
    with col2:
        st.metric("画面亮度", int(analysis['brightness']))
    with col3:
        st.metric("动态强度", int(analysis['motion']))
    
    # 预测得分
    predictor = SimplePredictor()
    score = predictor.predict(analysis)
    
    # 显示评分
    st.subheader("🔥 爆款指数")
    st.progress(score/100)
    st.write(f"综合评分：{score:.1f}分（满分100）")
    
    # 优化建议
    st.subheader("💡 优化指南")
    for suggestion in get_suggestions(analysis):
        st.error(suggestion)
    
    # 发布时间建议
    st.subheader("⏰ 最佳发布时间")
    st.success("推荐发布时间：晚上7点-9点（根据平台用户活跃时段）")

st.markdown("---")
st.caption("提示：本结果为模拟预测，实际效果需结合内容质量")