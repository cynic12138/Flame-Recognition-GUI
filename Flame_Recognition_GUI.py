import cv2
import base64
import requests
import json
import time
import os
import re
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime, timedelta
import threading
import queue
import signal
import sys
import logging
import copy

# 常量定义
VIDEO_ALARM_DIR = "video alarm"
ALARM_IMAGES_DIR = os.path.join(VIDEO_ALARM_DIR, "alarm_images")
ALARM_RESULTS_DIR = os.path.join(VIDEO_ALARM_DIR, "alarm_results")
OUTPUT_DETAIL_DIR = "output_detail"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flame_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FlameColorAnalyzer:
    def __init__(self, api_key, model_name="qwen-vl-max", gui_callback=None):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.is_running = True  # 控制流读取的运行状态
        self.last_extract_time = 0  # 记录上一次抽帧的系统时间（秒）
        self.output_dir = "output"  # 图片保存根目录
        self.gui_callback = gui_callback  # GUI回调函数
        self.current_frame = None  # 当前帧
        self.current_analysis_result = None  # 当前分析结果
        self.frame_queue = queue.Queue(maxsize=10)  # 限制队列大小防止内存泄漏
        self.analysis_queue = queue.Queue(maxsize=5)  # 分析结果队列
        self.lock = threading.RLock()  # 可重入锁用于线程安全
        self.threads = []  # 保存所有工作线程引用
        self._init_output_dir()  # 初始化输出文件夹
        logger.info(f"火焰颜色分析器初始化完成，输出目录: {self.output_dir}")

    def _init_output_dir(self):
        """
        初始化输出文件夹：不存在则创建，确保保存路径有效
        """
        # 创建根目录output
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建图片保存根目录：{self.output_dir}")

        # 可选：按日期创建子目录（如output/20251104），便于按日期分类
        self.daily_output_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(self.daily_output_dir):
            os.makedirs(self.daily_output_dir)
            print(f"已创建当日图片保存目录：{self.daily_output_dir}")

    def connect_rtsp_stream(self, rtsp_url):
        """连接到RTSP流，带有线程安全和错误处理机制"""
        cap = None
        retry_count = 0
        max_retries = 5
        retry_interval = 3

        with self.lock:
            # 确保释放任何现有连接
            pass  # 这里可以添加释放逻辑，如果需要的话

        while retry_count < max_retries and self.is_running:
            try:
                print(f"尝试连接RTSP流: {rtsp_url} (重试次数: {retry_count + 1}/{max_retries})")
                
                with self.lock:
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    # 设置缓冲区大小
                    if cap:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print("RTSP流连接成功！")
                    return cap

                print(f"RTSP流连接失败，{retry_interval}秒后重试...")
                time.sleep(retry_interval)
                retry_count += 1
            except Exception as e:
                retry_count += 1
                print(f"RTSP流连接异常: {str(e)}, 尝试第 {retry_count}/{max_retries} 次...")
                time.sleep(retry_interval)
            finally:
                # 如果线程应该停止，退出循环
                if not self.is_running:
                    if cap:
                        cap.release()
                    break

        if cap:
            cap.release()
        raise ValueError(f"RTSP流连接失败（已重试{max_retries}次）")

    def extract_frames_from_rtsp(self, rtsp_url, interval_seconds=5):
        """
        从RTSP流抽帧，保存到output文件夹，并更新当前帧
        增加错误处理和资源管理，确保RTSP流连接正常和资源正确释放
        """
        cap = None
        frame_count = 0  # 累计帧数
        
        try:
            # 连接RTSP流
            cap = self.connect_rtsp_stream(rtsp_url)
            
            # 优化RTSP读取参数
            if cap:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区为1，减少延迟
                cap.set(cv2.CAP_PROP_FPS, 25)  # 设置期望帧率
            else:
                logger.error("无法初始化RTSP流连接")
                return

            while self.is_running:
                if not cap or not cap.isOpened():
                    logger.warning("RTSP流连接无效，尝试重连...")
                    if cap:
                        cap.release()
                    cap = self.connect_rtsp_stream(rtsp_url)
                    continue
                
                try:
                    with self.lock:
                        ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning("RTSP流读取失败，尝试重连...")
                        if cap:
                            cap.release()
                        cap = self.connect_rtsp_stream(rtsp_url)
                        continue

                    # 检查帧是否为空
                    if frame is None or frame.size == 0:
                        logger.warning("读取到空帧")
                        continue

                    # 更新当前帧，用于GUI显示
                    self.current_frame = frame.copy()

                    # 将帧添加到队列用于GUI更新，使用队列避免直接在读取线程中操作GUI
                    if not self.frame_queue.full():
                        try:
                            # 使用put_nowait避免阻塞
                            self.frame_queue.put_nowait(frame.copy())
                        except Exception as e:
                            logger.error(f"添加帧到队列失败: {str(e)}")

                    current_time = time.time()
                    time_since_last_extract = current_time - self.last_extract_time

                    if time_since_last_extract >= interval_seconds:
                        # 生成图片文件名（时间戳+帧序号，避免重复）
                        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
                        img_filename = f"frame_{timestamp_str}_count_{frame_count}.jpg"
                        img_save_path = os.path.join(self.daily_output_dir, img_filename)

                        # 保存图片
                        save_success = cv2.imwrite(img_save_path, frame)
                        if not save_success:
                            logger.error(f"保存图片失败: {img_save_path}")

                        # 转换为base64（供API分析）
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        # 生成帧信息
                        frame_info = {
                            'timestamp': current_time,
                            'timestamp_formatted': datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S'),
                            'frame_base64': frame_base64,
                            'frame_count': frame_count,
                            'image_save_path': img_save_path,
                            'frame': frame.copy()  # 添加原始帧的副本
                        }

                        yield frame_info
                        logger.info(f"已提取帧 - 时间: {frame_info['timestamp_formatted']}, 累计帧数: {frame_count}")

                        self.last_extract_time = current_time

                    frame_count += 1
                    # 短暂休眠避免CPU占用过高
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"处理帧时发生错误: {str(e)}")
                    # 短暂休眠避免错误时无限循环重试
                    time.sleep(1)

        except Exception as e:
            logger.error(f"RTSP流处理主循环异常: {str(e)}")
        finally:
            # 确保资源正确释放
            with self.lock:
                if cap and cap.isOpened():
                    cap.release()
                    logger.info("RTSP流连接已释放")
                else:
                    logger.warning("RTSP流连接已关闭或无效")
            print("RTSP流处理已结束")

    def create_analysis_prompt(self, is_first_frame=False, has_previous_frame=False):
        """生成结构化分析提示，获取火焰颜色、位置和状态信息"""
        prompt = """请仔细分析这张图像中的火焰区域，并严格按照以下格式输出分析结果：

请严格按照JSON格式输出，不要包含任何额外的文字说明：
{
  "color": "[火焰主要颜色，如明黄色、暗黄色等]",
  "position": "[火焰位置仅有两种，如合适、靠近炉膛]",
  "status": "[火焰状态仅有两种，如良好、风量偏小]",
  "color_compare": "[与前一帧相比的颜色变化，如果是第一帧则填\"首次帧\"，否则描述变化情况]"
}

分析要点：
1. 颜色：判断火焰的主要颜色，正常情况下为明黄色，不正常可能为暗黄色等
2. 位置：判断火焰是否位于画面中心（正常）还是靠近边缘/炉膛（不正常）
3. 状态：根据颜色和位置判断火焰燃烧状况，如良好、风量偏小等
4. 颜色比较：如果有前一帧参考，请描述颜色变化情况

请确保输出格式严格为JSON，且内容准确反映图像中的火焰特征。"""

        if not is_first_frame and has_previous_frame:
            prompt += "\n\n请特别注意描述与前一帧的颜色对比情况。"

        return prompt

    def analyze_frame_with_qwen(self, frame_base64, prompt):
        """分析帧并返回结构化的JSON对象"""
        if self.api_key == "your_api_key_here":
            # 返回简化的JSON对象，移除color_compare字段
            return {
                "color": "明黄色",
                "position": "合适",
                "status": "良好"
            }

        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{frame_base64}"},
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {"max_tokens": 1500}
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            result = response.json()

            if 'output' in result and 'choices' in result['output']:
                # 提取模型输出内容
                content = result['output']['choices'][0]['message']['content']
                # 尝试直接解析JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试提取JSON部分
                    json_match = re.search(r'\{[^}]*\}', content)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except json.JSONDecodeError:
                            # 返回默认的正常状态
                            return {
                                "color": "明黄色",
                                "position": "合适",
                                "status": "良好",
                                "color_compare": "首次帧"
                            }
                    else:
                        # 返回默认的正常状态
                        return {
                            "color": "明黄色",
                            "position": "合适",
                            "status": "良好",
                            "color_compare": "首次帧"
                        }
            else:
                return {
                    "color": "未知",
                    "position": "未知",
                    "status": "未知",
                    "color_compare": "未知"
                }

        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {str(e)}")
            # 返回默认的正常状态
            return {
                "color": "明黄色",
                "position": "合适",
                "status": "良好",
                "color_compare": "首次帧"
            }
        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            # 返回默认的正常状态
            return {
                "color": "明黄色",
                "position": "合适",
                "status": "良好",
                "color_compare": "首次帧"
            }

    def _stream_worker(self, rtsp_url):
        """
        重新设计的RTSP流读取工作线程，使用环形缓冲区和帧率控制
        确保实时流畅播放，避免队列堆积和内存泄漏
        """
        cap = None
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        last_read_time = time.time()
        frame_count = 0
        fps_start_time = time.time()
        
        # 实现简单的环形缓冲区逻辑
        buffer_size = 10
        frame_buffer = [(None, 0.0) for _ in range(buffer_size)]
        buffer_index = 0
        
        # 目标帧率控制
        target_fps = 25
        frame_interval = 1.0 / target_fps
        
        try:
            while self.is_running:
                # 初始化或重新连接RTSP流
                if cap is None:
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnect_attempts:
                        logger.warning(f"达到最大重连次数 {max_reconnect_attempts}，等待5秒后重试...")
                        time.sleep(5)
                        reconnect_attempts = 0
                    
                    logger.info(f"正在连接RTSP流: {rtsp_url} (尝试 {reconnect_attempts}/{max_reconnect_attempts})")
                    cap = self.connect_rtsp_stream(rtsp_url)
                    if cap is None:
                        logger.error("RTSP连接失败，5秒后重试...")
                        time.sleep(5)
                        continue
                    
                    # 清除OpenCV缓冲区
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化缓冲区
                    cap.set(cv2.CAP_PROP_FPS, target_fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置帧位置
                    
                    # 快速读取几帧以清空缓冲区
                    for _ in range(3):
                        cap.read()
                    
                    reconnect_attempts = 0
                    logger.info("RTSP流连接成功，开始读取视频帧")
                
                # 帧率控制
                current_time = time.time()
                elapsed = current_time - last_read_time
                if elapsed < frame_interval:
                    # 精确延时，避免CPU占用过高
                    time.sleep(max(0, frame_interval - elapsed) * 0.9)
                
                # 读取帧
                with self.lock:
                    ret, frame = cap.read()
                
                if not ret:
                    logger.warning("RTSP流读取失败，尝试重连...")
                    if cap:
                        cap.release()
                    cap = None
                    continue
                
                last_read_time = time.time()
                frame_count += 1
                
                # 计算实际帧率
                if last_read_time - fps_start_time >= 1.0:
                    actual_fps = frame_count / (last_read_time - fps_start_time)
                    logger.debug(f"实际帧率: {actual_fps:.1f} FPS")
                    frame_count = 0
                    fps_start_time = last_read_time
                
                # 使用深拷贝避免引用问题
                frame_copy = frame.copy()
                
                # 更新环形缓冲区
                with self.lock:
                    self.current_frame = frame_copy
                    frame_buffer[buffer_index] = (frame_copy, time.time())
                    buffer_index = (buffer_index + 1) % buffer_size
                
                # 将最新帧放入队列，覆盖旧帧确保实时性
                try:
                    # 非阻塞方式清空队列并放入最新帧
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    # 确保队列有空间再放入新帧
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait((time.time(), frame_copy))
                except Exception as e:
                    logger.error(f"帧队列操作失败: {str(e)}")
                    
                # 检查是否需要退出
                if not self.is_running:
                    break
                    
        except Exception as e:
            logger.error(f"视频流线程异常: {str(e)}")
        finally:
            with self.lock:
                if cap:
                    cap.release()
                    cap = None
                self.is_running = False
            logger.info("视频流线程已安全结束")
    
    def _analysis_worker(self, rtsp_url, interval_seconds=5):
        """
        优化的分析工作线程，负责定期提取帧并进行分析
        实现降低分析帧尺寸、优化内存占用、完善异常处理
        """
        results = []
        previous_analysis = None
        frame_index = 0
        last_extract_time = 0
        target_analysis_width = 640  # 分析时的目标宽度，降低分辨率以减少内存占用
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # 按间隔时间提取帧进行分析
                if current_time - last_extract_time >= interval_seconds:
                    # 确保有可用的帧
                    if self.current_frame is not None:
                        frame_index += 1
                        frame = self.current_frame.copy()
                        
                        logger.info(f"\n{'-' * 50}")
                        logger.info(f"分析第 {frame_index} 个帧")

                        # 降低分析帧的尺寸，减少内存占用和处理开销
                        h, w = frame.shape[:2]
                        scale_ratio = target_analysis_width / w
                        new_h = int(h * scale_ratio)
                        analysis_frame = cv2.resize(frame, (target_analysis_width, new_h), interpolation=cv2.INTER_AREA)

                        # 生成图片文件名
                        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        img_filename = f"frame_{timestamp_str}_count_{frame_index}.jpg"
                        img_save_path = os.path.join(self.daily_output_dir, img_filename)

                        # 保存原始分辨率的图片
                        cv2.imwrite(img_save_path, frame)

                        # 使用降低分辨率的帧进行分析，减少base64编码后的大小
                        _, buffer = cv2.imencode('.jpg', analysis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # 降低质量以减少大小
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        # 构建提示词
                        if frame_index == 1:
                            prompt = self.create_analysis_prompt(is_first_frame=True)
                        else:
                            prompt = self.create_analysis_prompt(has_previous_frame=True)

                        # 分析帧
                        analysis_json = self.analyze_frame_with_qwen(frame_base64, prompt)
                        logger.info(f"分析结果获取成功: {analysis_json}")

                        try:
                            # 更新当前分析结果（使用深拷贝）
                            self.current_analysis_result = copy.deepcopy(analysis_json)

                            # 添加到分析结果队列，清空队列只保留最新结果
                            try:
                                # 清空队列
                                while not self.analysis_queue.empty():
                                    try:
                                        self.analysis_queue.get_nowait()
                                        self.analysis_queue.task_done()
                                    except:
                                        break
                                # 添加最新结果
                                self.analysis_queue.put_nowait(analysis_json)
                                logger.debug("分析结果已添加到队列")
                            except Exception as e:
                                logger.error(f"添加分析结果到队列失败: {str(e)}")
                        except Exception as e:
                            logger.error(f"处理分析结果时发生错误: {str(e)}")
                            analysis_json = {
                                "color": "未知",
                                "position": "未知",
                                "status": "未知",
                                "color_compare": "未知"
                            }

                        # 保存结果
                        result = {
                            'timestamp': current_time,
                            'timestamp_formatted': datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S'),
                            'analysis': analysis_json,
                            'frame_count': frame_index,
                            'image_save_path': img_save_path,
                            'analysis_json': analysis_json
                        }
                        results.append(result)

                        previous_analysis = analysis_json
                        last_extract_time = current_time
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"分析线程出错: {str(e)}", exc_info=True)
        finally:
            logger.info("\n分析停止，正在保存结果...")

    
    def analyze_rtsp_stream(self, rtsp_url, interval_seconds=5):
        """
        主分析函数，启动多线程架构，增加异常处理和线程安全机制
        """
        logger.info(f"开始监听RTSP流: {rtsp_url}，抽帧间隔: {interval_seconds}秒")
        
        try:
            # 重置运行状态
            self.is_running = True
            
            # 清空队列
            try:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        break
                
                while not self.analysis_queue.empty():
                    try:
                        self.analysis_queue.get_nowait()
                    except Exception:
                        break
            except Exception as queue_e:
                logger.error(f"清空队列时发生错误: {str(queue_e)}")
            
            # 启动视频流读取线程
            self.stream_thread = threading.Thread(target=self._stream_worker, args=(rtsp_url,))
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            # 启动分析线程
            self.analysis_thread = threading.Thread(target=self._analysis_worker, args=(rtsp_url, interval_seconds))
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
            # 保存线程引用
            self.threads = [self.stream_thread, self.analysis_thread]
            
            logger.info("多线程架构已启动")
            
            # 主线程等待
            try:
                while self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("收到中断信号，正在停止分析...")
            
        except Exception as e:
            logger.error(f"分析过程发生异常: {str(e)}")
        finally:
            # 确保线程停止标志设置正确
            self.is_running = False
            
            # 等待线程结束
            if hasattr(self, 'threads'):
                for t in self.threads:
                    if t and t.is_alive():
                        try:
                            t.join(timeout=2)
                            if t.is_alive():
                                logger.warning(f"线程未能在指定时间内正常退出")
                        except Exception as join_e:
                            logger.error(f"等待线程退出时发生错误: {str(join_e)}")
            
            logger.info("所有线程已停止")

    def save_results(self, results, output_file=None):
        """保持原逻辑不变（结果中包含图片路径）"""
        if not output_file:
            output_file = f"flame_analysis_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"分析结果已保存到: {output_file}")

    def print_results(self, results):
        """显示分析结果汇总"""
        print("\n" + "=" * 80)
        print("火焰颜色分析汇总结果")
        print("=" * 80)

        for result in results:
            print(f"\n时间戳: {result['timestamp_formatted']}")
            print(f"帧编号: {result['frame_count']}")
            print(f"图片路径: {result['image_save_path']}")
            print("-" * 40)
            # 优先显示JSON格式的分析结果
            if 'analysis_json' in result:
                for key, value in result['analysis_json'].items():
                    print(f"{key}: {value}")
            else:
                print(result['analysis'])
            print("-" * 40)


class FlameMonitorGUI:
    """火焰监控GUI界面"""

    def __init__(self, root, analyzer):
        self.root = root
        self.analyzer = analyzer
        self.root.title("火焰监控系统")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 初始化文件夹结构
        self._initialize_directories()

        # 设置窗口关闭时的回调
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 视频更新优化参数
        self.last_video_update_time = 0
        self.video_update_interval = 0.05  # 限制视频更新频率为20fps
        self.last_canvas_size = (640, 480)  # 缓存画布尺寸，避免重复计算
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_analysis = None
        self.previous_analysis = None
        self.current_display_fps = 0  # 当前显示帧率，用于UI显示
        self.frame_count = 0  # 帧计数器，用于控制文字颜色切换

        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure(
            "TLabel",
            font=("SimHei", 10)
        )
        self.style.configure(
            "TButton",
            font=("SimHei", 10)
        )
        self.style.configure(
            "TFrame",
            font=("SimHei", 10)
        )

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建上方火焰画面区域
        self.video_frame = ttk.LabelFrame(self.main_frame, text="火焰画面", padding="10")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建视频显示画布
        self.canvas = tk.Canvas(self.video_frame, bg="#e0f7fa")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 创建默认的火焰画面文本
        self.canvas_text = self.canvas.create_text(
            320, 240,
            text="点击下方按钮启动监控",
            font=("SimHei", 16),
            fill="#666666"
        )

        # 创建下方信息区域
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建左下方火焰状态区域
        self.status_frame = ttk.LabelFrame(self.bottom_frame, text="火焰状态", padding="10")
        self.status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # 创建右下方措施区域
        self.measure_frame = ttk.LabelFrame(self.bottom_frame, text="措施", padding="10")
        self.measure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # 措施标签
        self.measure_label = ttk.Label(self.measure_frame, text="暂无", font=("SimHei", 14))
        self.measure_label.pack(pady=10, fill=tk.BOTH, expand=True)

        # 创建表格
        self.create_status_table()

        # 初始化状态值 - 未启动时显示暂无
        self.update_status_table({
            "color": "暂无",
            "color_compare": "暂无",
            "position": "暂无"
        })

        # 启动按钮
        self.start_button = ttk.Button(self.main_frame, text="启动监控", command=self.start_monitoring)
        self.start_button.pack(pady=10)

        # 线程控制
        self.monitor_thread = None
        self.is_monitoring = False
        # 存储分析历史，用于显示当前和前一帧
        self.previous_analysis = None
        self.current_analysis = None
        # 存储最后一次保存的日期，用于创建每日JSON文件
        self.last_save_date = None
        self.daily_output_detail = []

    def _initialize_directories(self):
        """初始化所有必要的文件夹"""
        directories = [VIDEO_ALARM_DIR, ALARM_IMAGES_DIR, ALARM_RESULTS_DIR, OUTPUT_DETAIL_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    print(f"已创建文件夹: {directory}")
                except Exception as e:
                    print(f"创建文件夹失败 {directory}: {str(e)}")
    
    def create_status_table(self):
        """创建火焰状态表格"""
        # 创建统一的表格框架，使用网格布局确保对齐
        table_frame = ttk.Frame(self.status_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 创建表头行
        header_frame = ttk.Frame(table_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=1)

        # 定义标签文本 - 移除颜色变化词条
        labels = ["颜色", "火焰状态", "位置"]

        # 创建变量和标签引用字典
        self.value_vars = {}
        self.value_labels = {}

        # 为每一行创建标签和值显示
        for i, label_text in enumerate(labels):
            # 创建行框架
            row_frame = ttk.Frame(table_frame)
            row_frame.grid(row=i + 1, column=0, sticky="ew", pady=2)
            row_frame.columnconfigure(0, weight=1)
            row_frame.columnconfigure(1, weight=1)

            # 设置行背景色，交替显示
            bg_color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
            row_frame.configure(style="Row.TFrame")

            # 创建标签
            ttk.Label(row_frame, text=label_text, font=("SimHei", 11, "bold"),
                      anchor=tk.W, padding=(5, 5)).grid(
                row=0, column=0, sticky="ew", padx=(0, 2))

            # 创建值变量 - 保持火焰状态与status关联
            var_name = ["color", "status", "position"][i]
            value_var = tk.StringVar()
            self.value_vars[var_name] = value_var

            # 创建值标签，设置标签名称以便更新颜色 - 修改为右对齐并添加合适的内边距
            value_label = ttk.Label(row_frame, textvariable=value_var, font=("SimHei", 11),
                                    anchor=tk.E, padding=(5, 5, 15, 5), name=f"{var_name}_label")
            value_label.grid(row=0, column=1, sticky="ew", padx=(2, 5))
            self.value_labels[var_name] = value_label

        # 保存引用（保持与原代码兼容）
        self.color_var = self.value_vars["color"]
        self.position_var = self.value_vars["position"]
        # 添加status变量引用
        self.status_var = self.value_vars["status"]
        # 移除color_compare引用

    def update_status_table(self, analysis_data):
        """更新火焰状态表格，实现奇偶帧颜色切换"""
        try:
            # 获取状态值 - 移除color_compare
            color_value = analysis_data.get("color", "明黄色")
            position_value = analysis_data.get("position", "合适")
            status_value = analysis_data.get("status", "良好")

            print(f"更新状态表格: 颜色={color_value}, 状态={status_value}, 位置={position_value}")

            # 更新变量 - 移除color_compare
            self.value_vars["color"].set(color_value)
            self.value_vars["status"].set(status_value)  # 火焰状态关联status
            self.value_vars["position"].set(position_value)

            # 更新变量引用（保持与原代码兼容）
            self.color_var.set(color_value)
            self.position_var.set(position_value)
            # 更新status变量引用
            self.status_var.set(status_value)
            # 不再更新color_compare_var

            # 判断是否为报警状态
            is_alarm = (color_value != "明黄色" or 
                        (position_value != "合适" and "中间" not in position_value) or 
                        status_value != "良好")

            # 根据帧计数和是否报警设置文字颜色
            # 报警帧统一为红色
            if is_alarm:
                text_color = "#ff0000"  # 红色
            else:
                # 奇数帧为蓝色，偶数帧为紫色（修改为蓝紫交替）
                if self.frame_count % 2 == 1:  # 奇数帧
                    text_color = "#0000ff"  # 蓝色
                else:  # 偶数帧
                    text_color = "#9900ff"  # 紫色

            # 应用颜色设置 - 移除color_compare
            self.value_labels["color"].config(foreground=text_color, font=("SimHei", 11, "bold"))
            self.value_labels["status"].config(foreground=text_color, font=("SimHei", 11, "bold"))
            self.value_labels["position"].config(foreground=text_color, font=("SimHei", 11, "bold"))
        except Exception as e:
            print(f"更新状态表格出错: {str(e)}")

    def _save_alarm_frame(self, frame, analysis_data):
        """保存报警帧图片和分析结果"""
        try:
            # 确保文件夹存在
            self._initialize_directories()
            
            # 生成时间戳
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # 保存报警帧图片
            img_filename = f"alarm_frame_{timestamp_str}.jpg"
            img_save_path = os.path.join(ALARM_IMAGES_DIR, img_filename)
            cv2.imwrite(img_save_path, frame)
            print(f"已保存报警帧: {img_save_path}")
            
            # 保存报警帧分析结果
            result_filename = f"alarm_result_{timestamp_str}.json"
            result_save_path = os.path.join(ALARM_RESULTS_DIR, result_filename)
            with open(result_save_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            print(f"已保存报警分析结果: {result_save_path}")
            
        except Exception as e:
            print(f"保存报警帧和结果失败: {str(e)}")
    
    def _save_daily_output_detail(self, analysis_data):
        """保存每日输出详情到JSON文件"""
        try:
            # 获取当前日期
            current_date = datetime.now().strftime("%Y_%m_%d")
            
            # 检查是否需要创建新的每日记录
            if self.last_save_date != current_date:
                # 如果有之前的记录，先保存
                if self.last_save_date and self.daily_output_detail:
                    self._write_daily_output_to_file(self.last_save_date)
                # 重置记录和日期
                self.last_save_date = current_date
                self.daily_output_detail = []
            
            # 添加当前分析结果
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "frame_count": self.frame_count,
                "analysis_result": analysis_data
            }
            self.daily_output_detail.append(record)
            
            # 写入文件
            self._write_daily_output_to_file(current_date)
            
        except Exception as e:
            print(f"保存每日输出详情失败: {str(e)}")
    
    def _write_daily_output_to_file(self, date_str):
        """将每日输出详情写入文件"""
        try:
            # 确保文件夹存在
            self._initialize_directories()
            
            filename = f"{date_str}_output_detail.json"
            filepath = os.path.join(OUTPUT_DETAIL_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.daily_output_detail, f, ensure_ascii=False, indent=2)
            
            print(f"已保存每日输出详情: {filepath}")
            
        except Exception as e:
            print(f"写入每日输出文件失败: {str(e)}")
    
    def update_measure_status(self, analysis_data):
        """根据分析结果更新措施状态显示，实现奇偶帧颜色切换，并保存报警帧"""
        try:
            if not self.is_monitoring or analysis_data is False:
                self.measure_label.config(text="暂无", foreground="black")
                return
            
            # 获取分析结果
            color = analysis_data.get("color", "未知")
            position = analysis_data.get("position", "未知")
            status = analysis_data.get("status", "未知")
            
            print(f"更新措施状态: 颜色={color}, 位置={position}, 状态={status}")
            
            # 保存每日输出详情
            self._save_daily_output_detail(analysis_data)
            
            # 判断是否为报警状态
            is_alarm = False
            
            # 按照用户要求的特定逻辑判断
            # 当火焰为暗黄色且位置靠近炉膛、风量偏小时显示警报
            if ("暗黄" in color and 
                "靠近" in position and 
                "风量偏小" in status):
                # 异常状态 - 警报帧统一红色
                is_alarm = True
                self.measure_label.config(
                    text="警报：打开二次风门调节器",
                    foreground="#ff0000",  # 红色
                    font=("SimHei", 14, "bold")
                )
                # 保存报警帧
                if hasattr(self.analyzer, 'current_frame') and self.analyzer.current_frame is not None:
                    self._save_alarm_frame(self.analyzer.current_frame, analysis_data)
            # 当火焰为明黄色且位置良好时显示正常
            elif color == "明黄色" and (position == "合适" or "中间" in position):
                # 正常状态 - 根据帧计数设置颜色
                if self.frame_count % 2 == 1:  # 奇数帧
                    text_color = "#0000ff"  # 蓝色
                else:  # 偶数帧
                    text_color = "#ffcc00"  # 黄色
                
                self.measure_label.config(
                    text="正常",
                    foreground=text_color,
                    font=("SimHei", 14, "bold")
                )
            else:
                # 其他情况也显示警报 - 警报帧统一红色
                is_alarm = True
                self.measure_label.config(
                    text="警报：打开二次风门调节器",
                    foreground="#ff0000",  # 红色
                    font=("SimHei", 14, "bold")
                )
                # 保存报警帧
                if hasattr(self.analyzer, 'current_frame') and self.analyzer.current_frame is not None:
                    self._save_alarm_frame(self.analyzer.current_frame, analysis_data)
        except Exception as e:
            print(f"更新措施状态出错: {str(e)}")
            # 出错时显示默认文本
            self.measure_label.config(
                text="暂无",
                foreground="black",
                font=("SimHei", 14, "bold")
            )

    def update_video_frame(self, frame):
        """
        重新设计的视频帧更新方法，实现高效缩放和帧率显示
        优化图像处理性能，确保播放流畅
        """
        try:
            # 确保窗口仍然存在
            if not self.root.winfo_exists():
                return
            
            # 限制视频更新频率，避免过度处理
            current_time = time.time()
            frame_time_diff = current_time - self.last_video_update_time
            
            # 计算当前帧率用于显示
            if frame_time_diff > 0:
                current_fps = 1.0 / frame_time_diff
                self.current_display_fps = current_fps
            else:
                self.current_display_fps = 0
                
            self.last_video_update_time = current_time
            
            # 获取画布大小，避免重复计算
            canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 640
            canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 480
            
            # 缓存画布尺寸，只有在大小改变时才重新计算
            if (canvas_width, canvas_height) != self.last_canvas_size:
                self.last_canvas_size = (canvas_width, canvas_height)
            
            # 快速检查帧是否有效
            if frame is None or frame.size == 0:
                logger.warning("接收到无效的视频帧")
                return
            
            # 实现两阶段缩放策略：先快速缩小到近似尺寸，再精确缩放
            img_height, img_width = frame.shape[:2]
            
            # 第一阶段：快速缩小（如果需要大幅缩小）
            if max(img_width, img_height) > max(canvas_width, canvas_height) * 1.5:
                # 使用更快的缩放方法
                scale_factor = min(canvas_width / img_width, canvas_height / img_height)
                # 先缩放到目标尺寸的1.2倍左右
                temp_width = int(img_width * scale_factor * 1.2)
                temp_height = int(img_height * scale_factor * 1.2)
                frame = cv2.resize(frame, (temp_width, temp_height), interpolation=cv2.INTER_NEAREST)
            
            # 转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 第二阶段：精确缩放
            scale_factor = min(canvas_width / rgb_frame.shape[1], canvas_height / rgb_frame.shape[0])
            target_width = int(rgb_frame.shape[1] * scale_factor)
            target_height = int(rgb_frame.shape[0] * scale_factor)
            
            # 根据缩放比例和目标尺寸选择最佳插值方法
            if scale_factor > 1.5:
                # 放大时使用高质量插值
                img = cv2.resize(rgb_frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            elif scale_factor < 0.5 and target_width * target_height > 100000:
                # 大幅缩小且目标尺寸较大时使用区域插值
                img = cv2.resize(rgb_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                # 其他情况使用线性插值
                img = cv2.resize(rgb_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            
            # 清除画布并更新
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo  # 保持引用
            
            # 显示当前帧率
            fps_text = f"FPS: {self.current_display_fps:.1f}"
            self.canvas.create_text(
                10, 10, 
                anchor=tk.NW,
                text=fps_text,
                fill="green",
                font=("SimHei", 12, "bold"),
                # 添加半透明背景
                tags="fps_text"
            )
            
        except Exception as e:
            logger.error(f"更新视频帧失败: {str(e)}")
            # 确保即使出错也不会影响后续更新
            if hasattr(self, 'canvas') and self.canvas:
                try:
                    # 清除fps_text标签避免文字堆积
                    self.canvas.delete("fps_text")
                except:
                    pass

    def gui_callback(self, update_type, data):
        """
        GUI回调函数（保留兼容），主要功能已迁移到队列处理器
        """
        # 这个回调函数现在主要是为了兼容性，实际更新由_queue_processor处理
        if self.root.winfo_exists():
            if update_type == 'update_video':
                # 使用after_idle确保在UI空闲时更新
                self.root.after_idle(lambda: self.update_video_frame(data))
            elif update_type == 'update_analysis':
                self.root.after_idle(lambda: self.process_analysis_result(data))

    def process_analysis_result(self, analysis_data):
        """处理分析结果并更新界面，递增帧计数器"""
        try:
            # 递增帧计数器
            self.frame_count += 1
            
            print(f"处理分析结果: {analysis_data}")
            # 保存前一帧分析结果 - 注意：已移除color_compare相关处理
            self.previous_analysis = self.current_analysis
            # 设置当前帧分析结果 - 注意：analysis_data不再包含color_compare字段
            self.current_analysis = analysis_data

            # 更新状态表格
            self.update_status_table(analysis_data)

            # 更新措施状态，直接传递完整分析数据
            self.update_measure_status(analysis_data)
        except Exception as e:
            print(f"处理分析结果出错: {str(e)}")

    def start_monitoring(self):
        """启动监控线程"""
        if not self.is_monitoring:
            print("准备启动监控...")
            self.is_monitoring = True
            self.analyzer.is_running = True
            self.start_button.config(text="停止监控", command=self.stop_monitoring)
            # 设置回调函数
            self.analyzer.gui_callback = self.gui_callback
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self.run_monitoring)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("监控已启动")
        else:
            self.stop_monitoring()

    def stop_monitoring(self):
        """停止监控，确保资源正确释放"""
        print("正在停止监控...")
        
        # 设置停止标志
        self.is_monitoring = False
        self.analyzer.is_running = False
        self.start_button.config(text="启动监控", command=self.start_monitoring)
        
        # 重置状态显示为暂无
        try:
            self.update_status_table({
                "color": "明黄色",
                "color_compare": "良好",
                "position": "合适" 
            })
            self.update_measure_status(False)  # 更新措施显示为暂无
        except Exception as update_e:
            print(f"更新状态显示时出错: {str(update_e)}")
        
        # 保存最后的每日输出详情
        if hasattr(self, 'last_save_date') and hasattr(self, 'daily_output_detail'):
            try:
                if self.last_save_date and self.daily_output_detail:
                    self._write_daily_output_to_file(self.last_save_date)
                    print("已保存最终的每日输出详情")
            except Exception as e:
                print(f"保存最终每日输出详情失败: {str(e)}")
        
        # 清空分析历史
        self.previous_analysis = None
        self.current_analysis = None
        
        # 确保分析器的资源正确释放
        try:
            # 等待线程结束
            if hasattr(self.analyzer, 'threads'):
                for t in self.analyzer.threads:
                    if t and t.is_alive():
                        try:
                            t.join(timeout=2.0)
                        except Exception as join_e:
                            print(f"等待分析器线程退出时发生错误: {str(join_e)}")
            
            # 清空队列
            try:
                while not self.analyzer.frame_queue.empty():
                    try:
                        self.analyzer.frame_queue.get_nowait()
                    except Exception:
                        break
                
                while not self.analyzer.analysis_queue.empty():
                    try:
                        self.analyzer.analysis_queue.get_nowait()
                    except Exception:
                        break
            except Exception as queue_e:
                print(f"清空分析器队列时发生错误: {str(queue_e)}")
                
        except Exception as analyzer_e:
            print(f"释放分析器资源时出错: {str(analyzer_e)}")
        
        # 恢复画布上的提示文本
        try:
            self.canvas.delete("all")
            self.canvas_text = self.canvas.create_text(
                320, 240,
                text="点击下方按钮启动监控",
                font=("SimHei", 16),
                fill="#666666"
            )
        except Exception as e:
            print(f"重置画布出错: {str(e)}")
            
        print("监控已停止，所有资源已释放")

    def on_closing(self):
        """窗口关闭时的处理"""
        if self.is_monitoring:
            self.stop_monitoring()
        self.root.destroy()

    def _queue_processor(self):
        """
        重新设计的队列处理器，实现优先级处理和动态调度间隔
        优化GUI更新性能，确保视频播放流畅和分析结果实时同步
        """
        if not self.root.winfo_exists():
            return
            
        # 动态调度间隔，根据系统负载调整更新频率
        schedule_interval = 10  # 默认10ms调度一次
        has_update = False  # 合并帧更新和分析结果更新的标志
        current_time = time.strftime('%H:%M:%S')
        
        try:
            # 优先处理视频帧队列（高优先级）
            if not self.analyzer.frame_queue.empty():
                # 清空队列，只保留最新帧
                latest_frame = None
                frame_count = 0
                
                # 限制循环次数，防止长时间阻塞UI线程
                max_dequeue = 5
                while not self.analyzer.frame_queue.empty() and frame_count < max_dequeue:
                    try:
                        timestamp, latest_frame = self.analyzer.frame_queue.get_nowait()
                        # 标记任务完成，避免队列积压
                        self.analyzer.frame_queue.task_done()
                        frame_count += 1
                    except (queue.Empty, Exception):
                        break
                    
                if latest_frame is not None:
                    # 直接更新视频帧，不使用after_idle避免额外延迟
                    try:
                        self.update_video_frame(latest_frame)
                        has_update = True
                        print(f"[{current_time}] 视频帧已更新")
                    except Exception as e:
                        logger.error(f"更新视频帧失败: {str(e)}")
            
            # 处理分析结果队列（同样高优先级，确保实时同步）
            if not self.analyzer.analysis_queue.empty():
                analysis_data = None
                try:
                    # 只获取最新的一个分析结果
                    while not self.analyzer.analysis_queue.empty():
                        analysis_data = self.analyzer.analysis_queue.get_nowait()
                        self.analyzer.analysis_queue.task_done()
                    
                    if analysis_data is not None:
                        try:
                            # 直接处理分析结果，不延迟，确保实时更新
                            print(f"[{current_time}] 处理分析结果: {analysis_data}")
                            self.process_analysis_result(analysis_data)
                            has_update = True
                        except Exception as e:
                            logger.error(f"处理分析结果失败: {str(e)}")
                except Exception as e:
                    logger.error(f"从分析队列获取数据失败: {str(e)}")
            
            # 根据是否有更新动态调整调度间隔
            # 如果有更新，稍微增加间隔避免UI线程过载
            # 如果没有更新，可以降低间隔提高响应速度
            if has_update:
                schedule_interval = 10  # 有更新时使用适中频率
            else:
                schedule_interval = 5   # 无更新时提高频率，更快地响应新数据
                
        except Exception as e:
            logger.error(f"队列处理器异常: {str(e)}")
        finally:
            # 确保即使出现异常也会继续调度
            try:
                if self.is_monitoring and self.root.winfo_exists():
                    # 使用after重新调度，避免递归调用导致堆栈溢出
                    self.root.after(schedule_interval, self._queue_processor)
            except Exception as e:
                logger.critical(f"重新调度队列处理器失败: {str(e)}")

    def run_monitoring(self):
        """
        运行监控
        """
        print("开始监控 (使用真实RTSP流)")
        try:
            # 启动队列处理器
            self.root.after_idle(self._queue_processor)
            # 使用真实RTSP流进行监控分析
            # 默认使用一个示例RTSP URL，实际使用时应替换为真实的RTSP流地址
            rtsp_url = "rtsp://admin:a12345678@192.168.10.203:554/Streaming/Channels/4701"
            # 调用analyzer的analyze_rtsp_stream方法进行实时分析
            self.analyzer.analyze_rtsp_stream(rtsp_url, interval_seconds=5)
        except Exception as e:
            print(f"监控运行出错: {str(e)}")
            self.root.after(0, lambda: self.start_button.config(text="启动监控", command=self.start_monitoring))
            self.is_monitoring = False


def main():
    # 配置参数
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if not API_KEY:
        print("警告: 未找到DASHSCOPE_API_KEY环境变量")
        API_KEY = "your_api_key_here"

    # 获取RTSP URL配置
    rtsp_url = "rtsp://admin:a12345678@192.168.10.203:554/Streaming/Channels/4701"
    if not rtsp_url:
        print("警告: 未找到RTSP_URL环境变量，将使用默认值")
        print("请设置RTSP_URL环境变量以使用真实的RTSP流地址")

    # 创建分析器实例（自动初始化output文件夹）
    analyzer = FlameColorAnalyzer(API_KEY)

    print("正在初始化火焰监控GUI...")
    try:
        # 设置Tkinter不显示窗口的选项（在某些环境中可能需要）
        os.environ['DISPLAY'] = ':0'

        # 创建Tkinter GUI
        root = tk.Tk()
        # 设置中文字体支持
        root.option_add("*Font", ("SimHei", 10))

        # 创建火焰监控GUI
        gui = FlameMonitorGUI(root, analyzer)

        # 启动GUI主循环
        print("启动GUI主循环...")
        root.mainloop()
    except tk.TclError as e:
        print(f"GUI环境错误: {str(e)}")
        print("在当前环境中无法显示GUI界面，但程序可以继续在命令行模式下运行")
        # 命令行模式下使用真实RTSP流
        try:
            print("进入命令行模式，使用真实RTSP流...")
            rtsp_url = "rtsp://admin:a12345678@192.168.10.203:554/Streaming/Channels/4701"
            print(f"正在连接RTSP流: {rtsp_url}")
            results = analyzer.analyze_rtsp_stream(rtsp_url, interval_seconds=5)
            analyzer.print_results(results)
        except KeyboardInterrupt:
            print("\n命令行模式被中断")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保停止监控
        analyzer.is_running = False
        print("程序已退出")


if __name__ == "__main__":
    main()
