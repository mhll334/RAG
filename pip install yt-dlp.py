import os
import yt_dlp

channel_url = 'https://www.youtube.com/channel/UCsUSffSkOgPT43rKqRuydIg'
output_folder = 'downloads'
os.makedirs(output_folder, exist_ok=True)

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(output_folder, '%(upload_date)s_%(title)s.%(ext)s'),
    'quiet': False,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
        'preferredquality': '192',
    }],
    'ignoreerrors': True,
    # optionally limit download count:
    # 'playlistend': 10,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([channel_url])
