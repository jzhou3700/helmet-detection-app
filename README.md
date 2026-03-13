"# 🛡️ 电动车乘客安全帽检测系统

基于YOLO深度学习模型的智能视频分析系统，能够自动检测电动车乘客是否正确佩戴安全帽。

## 📋 功能特性

- ✅ **Web前台界面** - 基于Streamlit的现代化Web界面
- ✅ **视频上传与处理** - 支持多种视频格式
- ✅ **实时检测标注** - 在视频上绘制检测框和标签
- ✅ **统计分析** - 汇总未佩戴头盔的人数和佩戴率
- ✅ **结果导出** - 支持下载检测结果视频和CSV报告
- ✅ **GPU加速** - 自动检测并使用GPU加速推理

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 至少4GB内存
- CUDA 11.0+ (可选)

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/jzhou3700/helmet-detection-app.git
cd helmet-detection-app
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行应用
```bash
streamlit run app.py
```

## 📖 使用说明

### 基本流程
1. 上传视频文件（支持MP4、AVI等）
2. 调整检测参数（可选）
3. 点击处理，等待完成
4. 查看统计结果和标注视频
5. 下载结果文件

## 📊 检测说明

### 输出指标
- 总检测行人数
- 未佩戴头盔人数
- 头盔佩戴率百分比
- 处理总帧数

### 视频标注
- 🟢 绿色框 = 正确佩戴头盔
- 🔴 红色框 = 未佩戴头盔

## 📁 项目结构

```
helmet-detection-app/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── .streamlit/
│   └── config.toml
├── detector/
│   ├── __init__.py
│   ├── yolo_detector.py
│   └── video_processor.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

## 📝 许可证

MIT License" 
