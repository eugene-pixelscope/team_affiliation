from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
# mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
#                                        split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
