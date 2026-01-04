from moviepy.editor import VideoFileClip, concatenate_videoclips

clips = [
    VideoFileClip("videos/random/rl-video-episode-0.mp4"),
    VideoFileClip("videos/halftrained/rl-video-episode-0.mp4"),
    VideoFileClip("videos/fullytrained/rl-video-episode-0.mp4"),
]

final = concatenate_videoclips(clips, method="compose")
final.write_videofile("videos/evolution.mp4", fps=30)
