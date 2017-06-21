source("const.R")

###
readRun <- function() {
	print("reading tweets...")
	twit_stats_iphone <- readText('tweets/_iphone/', 'iphone')
	twit_stats_galaxy <- readText('tweets/_galaxy/', 'galaxy')
	twit_stats_lumia <- readText('tweets/_lumia/', 'lumia')
	#twit_stats_aircel <- readText('tweets/_aircel/', 'aircel')
	#twits <- rbind(twit_stats_iphone, twit_stats_galaxy, twit_stats_lumia, twit_stats_aircel)
	twits <- rbind(twit_stats_iphone, twit_stats_galaxy, twit_stats_lumia)
	#head(twits)
	print(paste("read", nrow(twits), "tweets..."))
	print("running sentiment analysis...")
	twits_score_iphone <- score.sentiment(twit_stats_iphone$text, pos, neg, .progress='text') 
	twits_score_iphone$name <- 'iphone'
	twits_score_iphone$time <- twit_stats_iphone$created
	twits_score_galaxy <- score.sentiment(twit_stats_galaxy$text, pos, neg, .progress='text') 
	twits_score_galaxy$name <- 'galaxy'
	twits_score_galaxy$time <- twit_stats_galaxy$created
	twits_score_lumia <- score.sentiment(twit_stats_lumia$text, pos, neg, .progress='text') 
	twits_score_lumia$name <- 'lumia'
	twits_score_lumia$time <- twit_stats_lumia$created
	#twits_score_aircel <- score.sentiment(twit_stats_aircel$text, pos, neg, .progress='text')
	#twits_score_aircel$name <- 'aircel'
	#twits_score_aircel$time <- twit_stats_aircel$created
	#twits <- rbind(twits_score_iphone, twits_score_galaxy, twits_score_lumia, twits_score_aircel)
	twits <- rbind(twits_score_iphone, twits_score_galaxy, twits_score_lumia)
	twits$day <- substring(twits$time, 1,10)
	twits$hour <- substring(twits$time, 12, 13)
	return(twits)
}

getAllZoomLevels <- function(zooms) {
	map <- list()
	for (z in zooms) {
		map[[z]] <- get_map(location = "India", zoom = z)
	}
	return(map)
}

twits_trend <<- NULL
twits_loc <<- NULL
corpus <<- NULL
fit <<- NULL
mapImages <<- list()

runOnce <- function() {
  
  maps <- getAllZoomLevels(4:5) #maps <- getAllZoomLevels(4:10)
  save(maps, file="maps.rData")  
}

fetchDataAndSave <- function() {
  
  twits_trend <<- readRun()
  write.csv(twits_trend, "twits_trend.csv", row.names=FALSE)
}

runAnalysisAndSaveResults <- function() {

  twits_trend <<- read.csv("twits_trend.csv", stringsAsFactors=FALSE)
  twits_loc <<- readText('tweets/_loc/', 'India')
  #print(paste("Total", nrow(twits_trend), "tweets analysed"))
  
  corpus <<- getCorpus(twits_trend$text)
  save(corpus, file="corpus.rData")
  
  fit <<- getHierarchical(corpus)
  save(fit, file="fitted.rData")

  saveMapImages()
  saveWordCloud()
  
}

loadResults <- function() {
  for (device in c(brands)) {
    mapImages[[device]] <<- paste(device, "_loc.png", sep="")
  }
  load("corpus.rData")
  #load("maps.rData")
}

saveMapImages <- function() {
  
  for (ZOOM in 4:5) {
    map <- maps[[ZOOM]]
    mapImages[[ZOOM]] <<- paste("mapImage", ZOOM, ".png", sep="")
    #dev.copy(file="test.png",device=png, bg="white",  width=640, height=352) 
    graphics.off()
    p <- ggmap(map) + geom_point(aes(x=long, y=lat, colour=Score, size=Tweets), data=twits_loc) +
      scale_colour_gradient(low="red", high="green")
    #scale_colour_gradient2(low="red", high="green")
    ggsave(filename=mapImages[[ZOOM]], plot=p)
  }
}

saveWordCloud <- function() {
  png(filename="wordcloud.png")
  plotWordCloud(corpus)
  dev.off()  
}

twits_trend <- read.csv("twits_trend.csv", stringsAsFactors=FALSE)
twits_loc <- readText('tweets/_loc/', 'India')
load("fitted.rData")
loadResults()
