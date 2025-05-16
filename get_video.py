from pytube import Playlist, YouTube
import os
import sys
import subprocess
from yt_dlp import YoutubeDL

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def to_channel_url(handle_or_url: str) -> str:
    """Normalize a channel handle or URL to a full YouTube channel URL."""
    if handle_or_url.startswith("http"):
        return handle_or_url.rstrip("/")
    if handle_or_url.startswith("@"):
        return f"https://www.youtube.com/{handle_or_url}"
    # plain text like "TritonPoker"
    return f"https://www.youtube.com/@{handle_or_url}"

def fetch_playlists(channel_url: str):
    """
    Use yt_dlp in metadata‑only mode to pull the list of playlists
    shown under the channel’s “Playlists” tab.
    """
    opts = {
        "skip_download": True,
        "extract_flat": "in_playlist",  # don’t resolve every video yet
        "quiet": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"{channel_url}/playlists", download=False)
    playlists = []
    for entry in info.get("entries", []):
        # Each entry is a playlist stub with 'title'
        pid = entry.get("id")
        title = entry.get("title")
        if pid and title:
            playlists.append(
                {
                    "title": title,
                    "id": pid,
                    "url": f"https://www.youtube.com/playlist?list={pid}",
                }
            )
    return playlists

def choose_playlist(playlists):
    """Print playlists, prompt for a numeric choice, and return the selected dict."""
    print("\nAvailable playlists:\n")
    for idx, pl in enumerate(playlists):
        print(f"[{idx}] {pl['title']}") 
    while True:
        try:
            i = int(input("\nEnter the number of the playlist to download: "))
            if 0 <= i < len(playlists):
                return playlists[i]
        except ValueError:
            pass
        print("Invalid selection, try again.")

def download_playlist(playlist_url, outdir="downloads", resolution=720):
    """
    Call yt‑dlp to fetch every video in the playlist.
    The output template groups by playlist title → keeps files tidy.
    """
    format_str = f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]"
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--continue",                 # resume partially‑downloaded files
        "-f", format_str, 
        "-o",
        f"{outdir}/%(playlist_title)s/%(playlist_index)03d - %(title)s.%(ext)s",
        playlist_url,
    ]
    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", help="YouTube channel handle or URL (e.g. @PokerGO)")
    parser.add_argument("--resolution", type=int, default=720, help="Preferred resolution (e.g. 1080, 720, 480)")
    args = parser.parse_args()

    channel_url = to_channel_url(args.channel)
    print(f"\nFetching playlists for {channel_url} ...")
    playlists = fetch_playlists(channel_url)
    if not playlists:
        print("No playlists found (or channel hidden).")
        sys.exit(1)

    selected = choose_playlist(playlists)
    print(f"\nChosen playlist: {selected['title']}\n")

    download_playlist(selected["url"], resolution=args.resolution)    if len(sys.argv) < 2:
        print("Usage: python poker_playlist_downloader.py <channel_handle_or_url>")
        sys.exit(1)

    channel_arg = sys.argv[1]
    channel_url = to_channel_url(channel_arg)

    print(f"\nFetching playlists for {channel_url} ...")
    playlists = fetch_playlists(channel_url)
    if not playlists:
        print("No playlists found (or channel hidden).")
        sys.exit(1)

    selected = choose_playlist(playlists)
    print(f"\nChosen playlist: {selected['title']}\n")
    download_playlist(selected["url"])

if __name__ == "__main__":
    main()

