#!/usr/bin/env python3
"""
SpeechLLM 推理腳本
支援單輪對話和串流推理
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import time

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from speechllm.models.speechllm import SpeechLLM, SpeechLLMConfig
from speechllm.inference.engine import SpeechLLMInferenceEngine, InferenceConfig
from speechllm.codecs.vocab_manager import VocabManager
from speechllm.codecs.audio_tokenizer import AudioTokenizer


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config: dict) -> SpeechLLMConfig:
    """創建模型配置"""
    model_config = config['model']
    return SpeechLLMConfig(
        llm_model_name=model_config['llm_model_name'],
        freeze_llm=model_config['freeze_llm'],
        use_lora=model_config['use_lora'],
        lora_rank=model_config['lora_rank'],
        lora_alpha=model_config['lora_alpha'],
        whisper_model_name=model_config['whisper_model_name'],
        freeze_whisper=model_config['freeze_whisper'],
        num_query_tokens=model_config['num_query_tokens'],
        qformer_hidden_size=model_config['qformer_hidden_size'],
        qformer_num_layers=model_config['qformer_num_layers'],
        audio_transformer_layers=model_config['audio_transformer_layers'],
        audio_transformer_hidden_size=model_config['audio_transformer_hidden_size'],
        num_rvq_layers=model_config['num_rvq_layers'],
        codebook_size=model_config['codebook_size'],
        use_gradient_checkpointing=model_config['use_gradient_checkpointing'],
        mixed_precision=model_config['mixed_precision']
    )


def create_inference_config(config: dict) -> InferenceConfig:
    """創建推理配置"""
    inference_config = config['inference']
    return InferenceConfig(
        max_length=inference_config['max_length'],
        temperature=inference_config['temperature'],
        top_k=inference_config['top_k'],
        top_p=inference_config['top_p'],
        do_sample=inference_config['do_sample'],
        repetition_penalty=inference_config['repetition_penalty'],
        sample_rate=inference_config['sample_rate'],
        chunk_duration=inference_config['chunk_duration'],
        max_audio_length=inference_config['max_audio_length'],
        stream_chunk_size=inference_config['stream_chunk_size'],
        vad_threshold=inference_config['vad_threshold'],
        silence_threshold=inference_config['silence_threshold'],
        enable_duplex=inference_config['enable_duplex'],
        interrupt_threshold=inference_config['interrupt_threshold'],
        response_delay=inference_config['response_delay'],
        device=inference_config['device'],
        fp16=inference_config['fp16']
    )


def load_model(model_path: str, model_config: SpeechLLMConfig) -> SpeechLLM:
    """載入訓練好的模型"""
    print(f"載入模型: {model_path}")
    
    # 初始化模型
    model = SpeechLLM(model_config)
    
    # 載入權重
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        print("模型權重載入完成")
    else:
        print("警告: 未找到模型權重文件，使用隨機初始化的模型")
    
    return model


def single_turn_inference(
    engine: SpeechLLMInferenceEngine,
    input_text: str = None,
    input_audio: str = None,
    output_dir: str = "./outputs",
    mode: str = "auto"
):
    """單輪對話推理"""
    print("\n=== 單輪對話推理 ===")
    
    if input_text:
        print(f"輸入文字: {input_text}")
    if input_audio:
        print(f"輸入音訊: {input_audio}")
    
    # 生成回應
    start_time = time.time()
    response = engine.generate_response(
        input_text=input_text,
        input_audio=input_audio,
        mode=mode
    )
    end_time = time.time()
    
    print(f"推理時間: {end_time - start_time:.2f} 秒")
    
    # 顯示結果
    if "text" in response:
        print(f"回應文字: {response['text']}")
    
    if "audio" in response and response["audio"] is not None:
        # 保存音訊
        output_audio_path = os.path.join(output_dir, f"response_{int(time.time())}.wav")
        os.makedirs(output_dir, exist_ok=True)
        
        sf.write(
            output_audio_path,
            response["audio"],
            response.get("sample_rate", 16000)
        )
        print(f"回應音訊已保存: {output_audio_path}")
    
    return response


def interactive_chat(engine: SpeechLLMInferenceEngine):
    """互動式聊天"""
    print("\n=== 互動式聊天模式 ===")
    print("輸入 'quit' 退出，輸入 'clear' 清空歷史")
    print("支援的指令:")
    print("  text: <文字>  - 文字輸入")
    print("  audio: <路徑> - 音訊文件輸入")
    print("  mode: <模式>  - 設置模式 (TITO/AITO/TIAO/AIAO)")
    
    current_mode = "auto"
    
    while True:
        try:
            user_input = input("\n用戶: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                engine.clear_conversation_history()
                print("對話歷史已清空")
                continue
            elif user_input.startswith('mode:'):
                current_mode = user_input.split(':', 1)[1].strip()
                print(f"模式已設置為: {current_mode}")
                continue
            elif user_input.startswith('text:'):
                input_text = user_input.split(':', 1)[1].strip()
                response = engine.generate_response(
                    input_text=input_text,
                    mode=current_mode
                )
            elif user_input.startswith('audio:'):
                input_audio = user_input.split(':', 1)[1].strip()
                if os.path.exists(input_audio):
                    response = engine.generate_response(
                        input_audio=input_audio,
                        mode=current_mode
                    )
                else:
                    print(f"音訊文件不存在: {input_audio}")
                    continue
            else:
                # 預設為文字輸入
                response = engine.generate_response(
                    input_text=user_input,
                    mode=current_mode
                )
            
            # 顯示回應
            if "text" in response:
                print(f"助手: {response['text']}")
            
            if "audio" in response and response["audio"] is not None:
                print("助手: [音訊回應已生成]")
                # 這裡可以播放音訊或保存到文件
        
        except KeyboardInterrupt:
            print("\n聊天已中斷")
            break
        except Exception as e:
            print(f"處理輸入時出錯: {e}")


def streaming_inference(engine: SpeechLLMInferenceEngine):
    """串流推理"""
    print("\n=== 串流推理模式 ===")
    print("按 Ctrl+C 停止串流")
    
    def text_callback(text: str):
        """文字輸出回調"""
        print(f"助手: {text}")
    
    try:
        # 啟動串流
        engine.start_streaming(text_callback=text_callback)
        
        # 模擬音訊輸入
        print("模擬音訊輸入...")
        for i in range(10):
            # 生成隨機音訊塊
            audio_chunk = np.random.randn(1024).astype(np.float32) * 0.1
            engine.add_audio_chunk(audio_chunk)
            time.sleep(0.1)
        
        # 等待處理完成
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("串流已中斷")
    finally:
        engine.stop_streaming()


def benchmark_inference(
    engine: SpeechLLMInferenceEngine,
    num_runs: int = 10,
    input_text: str = "你好，請介紹一下自己。"
):
    """推理性能測試"""
    print(f"\n=== 推理性能測試 ({num_runs} 次) ===")
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        response = engine.generate_response(input_text=input_text, mode="TITO")
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        print(f"第 {i+1} 次: {inference_time:.3f} 秒")
    
    # 統計結果
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n性能統計:")
    print(f"  平均時間: {avg_time:.3f} ± {std_time:.3f} 秒")
    print(f"  最短時間: {min_time:.3f} 秒")
    print(f"  最長時間: {max_time:.3f} 秒")
    print(f"  吞吐量: {1/avg_time:.2f} 次/秒")


def main():
    parser = argparse.ArgumentParser(description="SpeechLLM 推理腳本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路徑"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路徑"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "interactive", "streaming", "benchmark"],
        default="interactive",
        help="推理模式"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        help="輸入文字（單輪模式）"
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        help="輸入音訊文件路徑（單輪模式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="輸出目錄"
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        choices=["auto", "TITO", "AITO", "TIAO", "AIAO"],
        default="auto",
        help="任務模式"
    )
    
    args = parser.parse_args()
    
    # 載入配置
    print(f"載入配置文件: {args.config}")
    config = load_config(args.config)
    
    # 創建配置物件
    model_config = create_model_config(config)
    inference_config = create_inference_config(config)
    
    print("配置載入完成")
    
    # 載入模型
    model = load_model(args.model_path, model_config)
    
    # 創建推理引擎
    print("創建推理引擎...")
    engine = SpeechLLMInferenceEngine(model, inference_config)
    
    # 顯示模型資訊
    model_info = engine.get_model_info()
    print("模型資訊:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 執行推理
    if args.mode == "single":
        single_turn_inference(
            engine=engine,
            input_text=args.input_text,
            input_audio=args.input_audio,
            output_dir=args.output_dir,
            mode=args.task_mode
        )
    elif args.mode == "interactive":
        interactive_chat(engine)
    elif args.mode == "streaming":
        streaming_inference(engine)
    elif args.mode == "benchmark":
        benchmark_inference(engine)
    
    print("推理完成")


if __name__ == "__main__":
    main()
