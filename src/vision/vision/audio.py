import subprocess
import yt_dlp

def play_youtube_music(query):
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Search YouTube Music via YouTube search
        result = ydl.extract_info(f"ytsearch1:{query}", download=False)
        video_url = result["entries"][0]["url"]

    # Play using mpv
    #subprocess.run(["mpv", "--no-video", video_url])
    subprocess.run([
            "mpv",
            "--no-video",
            "--ao=alsa",
            "--audio-device=alsa/plughw:2,0",
            "--af=dynaudnorm=g=12:f=250,volume=5.0",
            video_url
        ])

if __name__ == "__main__":
    song = input("Enter song name: ")
    play_youtube_music(song)