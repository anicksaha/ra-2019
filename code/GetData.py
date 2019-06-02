import pytube

links = open('links.txt','r')

for link in links:
	
	try:
		video = pytube.YouTube(link)
		print(video.title)
		#print(video.video_id)
		#print(video.age_restricted)
		#print(video.thumbnail_url)
		#print()

		stream = video.streams.first()
		stream.download('./data/')
	except:
		print(link)


