"""Validate Spotify track links for an official playback-only embed.

Does not fetch audio, metadata, tokens or embeddings from Spotify.
"""
import re
from urllib.parse import urlsplit

def spotify_track_id(value):
    value=value.strip()
    if value.startswith('spotify:track:'):
        track_id=value[len('spotify:track:'):]
    else:
        url=urlsplit(value)
        if url.scheme!='https' or url.netloc.lower()!='open.spotify.com':
            raise ValueError('Use Spotify Share > Copy Song Link (https://open.spotify.com/track/...).')
        match=re.fullmatch(r'/(?:intl-[a-z]{2}/)?track/([A-Za-z0-9]{22})/?',url.path)
        if not match:raise ValueError('Use a single song link, not a playlist, album or shortened link.')
        track_id=match.group(1)
    if not re.fullmatch(r'[A-Za-z0-9]{22}',track_id):
        raise ValueError('The Spotify track ID must contain 22 letters/numbers.')
    return track_id

def spotify_embed_url(value):
    return 'https://open.spotify.com/embed/track/'+spotify_track_id(value)
