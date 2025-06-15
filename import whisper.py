import whisper
import os

# 設定影片資料夾路徑
video_folder = "/Users/macbook/愛健康"  # 你的影片存放位置
output_folder = "/Users/macbook/Downloads/transcripts"  # 存放轉錄文字的資料夾

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 讀取所有影片清單
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# 載入 Whisper 模型
model = whisper.load_model("base")  # 可用 "tiny", "small", "medium", "large"

for video_file in video_files: 
      try: 
           video_path = os.path.join(video_folder, video_file)
           output_text_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.txt") 
     # 轉錄音訊
           result = model.transcribe(video_path) 
# 存成文字檔 
           with open(output_text_path, "w", encoding="utf-8") as f: 
              f.write(result["text"])
           print(f"轉錄完成: {video_file} -> {output_text_path}") 
except Exception as e:
           print(f"錯誤處理 {video_file}: {e}")