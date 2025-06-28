import os
import subprocess
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from opencc import OpenCC

# === 資料夾與模型設定 ===
base_folder = r"/home/tseng/桌面/csmu/icare"
videos_folder = os.path.join(base_folder, "videos")
output_folder = os.path.join(base_folder, "output")
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
cc = OpenCC("s2t")  # 簡轉繁

print("📦 載入語音模型中...")
model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

# === 要重試的播放清單 ===
playlist_links = [
    "https://www.youtube.com/watch?v=xwFqHXxQYJw&list=PLn4BRs6TsL8OCjHiJttO_YMawiQQjWCgH",
    "https://www.youtube.com/watch?v=ZgR_lYsWSTo&list=PLn4BRs6TsL8MpPWra4pN5ovZnPUDpcHjN",
    "https://www.youtube.com/watch?v=UfYl-wGiM7o&list=PLn4BRs6TsL8Ni-nmPnFPNxi9aCDq8aVRr&pp=0gcJCWMEOCosWNin",
    "https://www.youtube.com/watch?v=_XjfnIGDxPI&list=PLn4BRs6TsL8PNL8vdggbJkdUejZMoM4SQ",
    "https://www.youtube.com/watch?v=A1O6Pp3Tv00&list=PLn4BRs6TsL8P66OUIeVSMtbo6V_5Eswh4"
]

print(f"🔄 準備重新處理 {len(playlist_links)} 個下載失敗的播放清單...")

# === 統計用 ===
total_transcribed = 0
total_skipped = 0
failed_playlists = []

for idx, playlist_url in enumerate(playlist_links):
    print(f"\n📋 [{idx+1}/{len(playlist_links)}] 處理播放清單：{playlist_url}")
    existing_files = set(os.listdir(videos_folder))

    # yt-dlp 指令
    download_cmd = [
        "yt-dlp",
        "--yes-playlist",
        "--ignore-errors",
        "--restrict-filenames",
        "-x", "--audio-format", "wav",
        "--no-post-overwrites",
        "-o", os.path.join(videos_folder, f"%(playlist_index)s_%(title)s.%(ext)s"),
        playlist_url,
    ]

    try:
        result = subprocess.run(download_cmd, check=False, capture_output=True, text=True)
        all_files = set(os.listdir(videos_folder))
        new_audio_files = sorted([f for f in (all_files - existing_files) if f.endswith(".wav")])

        if result.returncode != 0:
            if new_audio_files:
                print("⚠️ yt-dlp 發生錯誤（但仍有下載成功影片），將繼續處理成功部分。")
                print(f"stderr：{result.stderr.strip().splitlines()[-1]}")
            else:
                print(f"❌ 完全無法下載播放清單：{playlist_url}")
                print(f"stderr：{result.stderr.strip().splitlines()[-1]}")
                failed_playlists.append(playlist_url)
                continue
    except Exception as e:
        print(f"❌ 非預期錯誤：{playlist_url}")
        print(f"🛠️ 錯誤內容：{e}")
        failed_playlists.append(playlist_url)
        continue

    print(f"🎧 本清單共新增 {len(new_audio_files)} 支影片，開始語音辨識...")

    for i, audio_file in enumerate(new_audio_files):
        print(f"📝 處理第 {i+1}/{len(new_audio_files)} 支影片：{audio_file}")
        audio_path = os.path.join(videos_folder, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        output_txt_path = os.path.join(output_folder, base_name + ".txt")

        if os.path.exists(output_txt_path):
            print(f"⚠️ 已存在，略過：{output_txt_path}")
            total_skipped += 1
            continue

        try:
            res = model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"])
            text = cc.convert(text)

            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ 已完成轉錄：{audio_file}")
            total_transcribed += 1
        except Exception as e:
            print(f"❌ 語音辨識錯誤：{audio_file}，錯誤訊息：{e}")

# === 統計報告 ===
print("\n✅ 重試任務完成！")
print(f"📑 成功轉錄：{total_transcribed} 支影片")
print(f"🟡 略過（已有 txt）：{total_skipped} 支影片")

if failed_playlists:
    print(f"\n❌ 以下播放清單仍然下載失敗（共 {len(failed_playlists)}）：")
    for link in failed_playlists:
        print(f" - {link}")
