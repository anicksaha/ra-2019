import pytube

link = "https://www.youtube.com/watch?v=iTqnRKuAljI"
video = pytube.YouTube(link)

print(video.title)
print(video.video_id)
print(video.age_restricted)
print(video.thumbnail_url)

print()


stream = video.streams.first()

stream.download('../data/')