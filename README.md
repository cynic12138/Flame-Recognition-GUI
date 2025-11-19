# 火焰监控系统

这是一个基于计算机视觉和AI模型的火焰监控系统，能够实时分析视频流中的火焰状态，并提供直观的GUI界面展示监控结果。

## 功能特点

### 实时监控
- 支持RTSP视频流连接和实时帧分析
- 基于AI模型（Qwen-VL）进行火焰颜色、位置和状态识别
- 周期性抽帧分析，可配置分析间隔

### 界面展示
- 直观的GUI界面，显示实时视频画面
- 火焰状态表格，展示颜色、位置和状态信息
- 奇偶帧文字颜色交替（蓝色/紫色），报警帧红色高亮
- 自动措施建议显示

### 数据存储
- 按日期自动创建输出目录结构
- 报警帧自动保存为图片
- 分析结果以JSON格式保存
- 每日输出详情记录

### 系统管理
- 完善的异常处理和错误日志记录
- 多线程设计，保证界面响应流畅
- 优雅启停和资源释放机制

## 技术栈

- **语言**: Python 3.x
- **GUI**: Tkinter
- **图像处理**: OpenCV, PIL
- **AI模型**: Qwen-VL (阿里云通义千问)
- **线程管理**: threading, queue
- **日志**: logging
- **数据处理**: json, numpy

## 安装要求

### 依赖包

请确保安装以下Python包：

```bash
pip install opencv-python
pip install pillow
pip install numpy
pip install requests
```

或者使用项目中的requirements.txt文件：

```bash
pip install -r requirements.txt
```

### 环境配置

- **API密钥**: 需要配置阿里云通义千问API密钥
- **RTSP流地址**: 需要提供有效的RTSP视频流地址
- **文件权限**: 确保程序有创建和写入文件的权限

## 使用方法

### 1. 配置API密钥

在运行程序前，需要在代码中配置阿里云通义千问API密钥：

```python
# 在main函数中设置
api_key = "your_api_key_here"  # 替换为实际API密钥
```

### 2. 运行程序

```bash
python fire_identity_gui.py
```

### 3. 操作界面

- 点击"启动监控"按钮开始实时监控
- 监控过程中会实时显示火焰状态信息
- 发生报警时，文字会变为红色，并且会自动保存报警帧
- 点击"停止监控"按钮结束监控

## 目录结构

程序运行时会自动创建以下目录结构：

```
mcp-demo/
├── video alarm/                   # 报警相关文件
│   ├── alarm_images/              # 报警帧图片
│   └── alarm_results/             # 报警结果JSON
├── output_detail/                 # 每日输出详情
├── output/                        # 输出目录
│   └── YYYYMMDD/                  # 按日期分类的子目录
├── fire_identity_gui.py           # 主程序文件
├── flame_monitor.log              # 日志文件
└── requirements.txt               # 依赖列表
```

## 自定义配置

### 分析间隔

可以在启动监控时调整帧分析间隔：

```python
interval_seconds = 5  # 默认5秒分析一次
```

### 输出目录

输出目录路径可以在代码中自定义：

```python
self.output_dir = "output"  # 可修改为其他路径
```

## 注意事项

1. 请确保网络连接稳定，特别是访问API服务时
2. API调用可能产生费用，请合理设置分析间隔
3. 长时间运行时请注意内存占用，程序已实现队列大小限制
4. RTSP连接失败时会自动重试，请检查RTSP地址是否正确
5. 如需修改火焰状态文字颜色，请在`update_status_table`方法中调整

## 故障排除

### 常见问题

1. **无法连接RTSP流**：检查RTSP地址和网络连接
2. **API调用失败**：检查API密钥和网络连接
3. **界面无响应**：可能是处理线程异常，请查看日志文件
4. **无法保存图片**：检查文件系统权限

### 日志查看

程序生成的日志文件`flame_monitor.log`包含详细的运行信息和错误日志，可以用于排查问题。

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请在GitHub仓库提交Issue。
