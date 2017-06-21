###
authTwitter <- function(consumerKey, consumerSecret) {
	options(RCurlOptions = list(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl")))
	reqURL <- "https://api.twitter.com/oauth/request_token"
	accessURL <- "https://api.twitter.com/oauth/access_token"
	authURL <- "https://api.twitter.com/oauth/authorize"
	twitCred <- OAuthFactory$new(consumerKey=consumerKey,consumerSecret=consumerSecret,requestURL=reqURL,accessURL=accessURL,authURL=authURL)
	twitCred$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))
	return(twitCred)
}

###
authFacebook <- function(ID) {
	winDialog(type = "ok", "Make sure you have already signed into Facebook.\n\nWhen  browser opens, please click 'Get Access Token' twice. You DO NOT need to select/check any boxes for a public feed.\n\n After pressing OK, swich over to your now open browser.")
	browseURL(paste("http://developers.facebook.com/tools/explorer/?method=GET&path=", ID, sep=""))
	accessToken <- winDialogString("When  browser opens, please click 'Get Access Token' twice and copy/paste token below", "")
	return(accessToken)
}

###
writeTweets <- function(keywords, suffix, ntweets) {
	twit_stats <- searchTwitter(keywords, n=ntweets, lang="en")
	twit_df <- do.call("rbind", lapply(twit_stats, as.data.frame))
	write.csv(twit_df, file = paste("tweets/", suffix, "/", gsub(':', '_', Sys.time()), ".csv", sep=""))
}

writeTweetsTime <- function(keywords, suffix, ntweets, from, to) {
	twit_stats <- searchTwitter(keywords, n=ntweets, lang="en", since=from, until=to)
	#twit_stats <- searchTwitter(keywords, n=ntweets, lang="en", since="2014-03-23", until="2014-03-24")
	twit_df <- do.call("rbind", lapply(twit_stats, as.data.frame))
	write.csv(twit_df, file = paste("tweets/", suffix, "/", gsub(':', '_', Sys.time()), ".csv", sep=""))
}

###
writeTweetsLoc <- function(keywords, suffix, ntweets, cities, miles = '100mi') { # cities are india cities
	locations <- india.cities[india.cities$name %in% cities, ]
	twits_df <- NULL
	for (i in 1:(nrow(locations))) {
		cit <- locations[i,]
		twit_stats <- searchTwitter(keywords, geocode=paste(cit$lat, cit$long, miles, sep=","))
		twit_stats_text <- sapply(twit_stats, function(x) x$getText())
		n <- length(twit_stats_text)
		twit_stats_text <- paste(twit_stats_text, collapse='.')
		sentiment = score.sentiment(twit_stats_text, pos, neg, .progress='text')
		rbind(twits_df, data.frame(Score=sentiment$score, Tweets=n)) -> twits_df
	}
	twits_df <- cbind(locations, twits_df)
	write.csv(twits_df, file = paste("tweets/", suffix, "/", gsub(':', '_', Sys.time()), ".csv", sep=""))
}

###
writeFBMsgs <- function(keywords, suffix, from) {
	fbmsg <- getURL(paste("https://graph.facebook.com/search?q=", gsub(' ', '+', keywords), "&type=post&since=", from, "&limit=99999999999999&access_token=", token, "", sep=""))
	fb_stats <- fromJSON(fbmsg)
	output.id <- sapply(fb_stats$data, function(x) x['from'])
	output.id <- sapply(output.id, function(x) x['id'])
	output.msg <- sapply(fb_stats$data, function(x) x['message'])
	output.name <- sapply(fb_stats$data, function(x) x['from'])
	output.name <- sapply(output.name, function(x) x['name'])
	output.time <- sapply(fb_stats$data, function(x) x['created_time'])
	fb_df <- NULL
	for (i in 1:(length(output.msg))) {
		if (output.msg[i] != "NULL") {
			info <- getURL(paste("https://graph.facebook.com/", output.id[i], sep=""))
			info <- fromJSON(info)
			gender <- unlist(info['gender'])
			if (is.null(gender)) { # | is.na(gender)) {
				gender <- 'None'
			}
			fb_df <- rbind(fb_df, c(unlist(output.msg[[i]]), unlist(output.name[[i]]), unlist(output.time[[i]]), gender))
		}
	}
	#fb_df <- as.data.frame(fb_df)
	names(fb_df) <- c('message', 'name', 'time', 'gender')
	write.csv(fb_df, file = paste("fb/", suffix, "/", Sys.Date(), ".csv", sep=""))
}

###
readText <- function(folder, prodname, rem_dup = TRUE) {
	files <- list.files(folder)
	txt <- NULL
	for (f in files) {
		tbl <- read.csv(paste(folder, f, sep=''))
		tbl$name <- prodname
		txt <- rbind(txt, tbl)
	}
	if (rem_dup) {
		txt <- txt[!duplicated(txt), ] # remove duplicates
	}
	return(txt)
}
