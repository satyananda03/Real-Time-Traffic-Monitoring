import subprocess  # Import library untuk menjalankan command line
# --- FUNGSI BARU UNTUK MENDAPATKAN URL STREAM ---
def get_youtube_live_url(youtube_url):
    """
    Menggunakan yt-dlp untuk mendapatkan URL stream HLS (m3u8) langsung dari YouTube Live.
    Ini jauh lebih andal daripada pafy.
    """
    print("Mencoba mendapatkan URL stream dengan yt-dlp...")
    try:
        # Menjalankan command 'yt-dlp -g -f best [URL]'
        result = subprocess.run(
            ["yt-dlp", "-g", "-f", "best", youtube_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True  # Akan melempar error jika command gagal
        )

        stream_url = result.stdout.strip()
        print(f"Berhasil mendapatkan URL stream: {stream_url}")
        return stream_url

    except FileNotFoundError:
        print("Error: 'yt-dlp' tidak ditemukan. Pastikan sudah terinstal dan ada di PATH sistem Anda.")
        print("Anda bisa menginstalnya dengan 'pip install yt-dlp'")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error saat menjalankan yt-dlp: {e.stderr.strip()}")
        return None
# ----------------------------------------------------