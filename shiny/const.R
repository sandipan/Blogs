###
#library(ROAuth)
#library(twitteR)
library(wordcloud)
library(tm)
#library(sentiment)
#library(tm.plugin.sentiment)
library(RColorBrewer)
library(RCurl)
library(rworldmap)
library(scales)
library(maps)
library(data.table)
library(ggplot2)
library(plyr)
library(stringr)
library(e1071)
library(RJSONIO)
library(ggmap)
library(mapproj)
library(rjson)
library(xts)
library(sqldf)
library(ggdendro)
library(rCharts)
data(world.cities)

###
#setwd("C:/Research/Quantta/shiny/sentiment")

###
source("credentials.R")
source("sentiment_algo.R")
source("social_media_api.R")
source("sentiment_plot.R")
source("hplot.R")

###
pos <- readLines("positive_words.txt")
neg <- readLines("negative_words.txt")

###
india.cities <- world.cities[world.cities$country.etc == "India",]

india_capitals <- c("Hyderabad","Itanagar","Dispur","Patna","Raipur","Panaji","Gandhinagar","Chandigarh","Shimla","Srinagar","Jammu","Ranchi","Bengaluru","Thiruvananthapuram","Bhopal","Bombay","Imphal","Shillong","Aizawl","Kohima","Bhubaneswar","Chandigarh","Jaipur","Gangtok","Madras","Agartala","Lucknow","Dehradun","Calcutta","Port Blair","Chandigarh","Silvassa","Daman","Delhi","Kavaratti","Pondicherry")

wb_major_cities <- c("Kolkata","Asansol","Bardhaman","Siliguri","Darjeeling","Jalpaiguri","Durgapur","Malda","Baharampur","Habra","Kharagpur","Shantipur",
"Dankuni","Dhulian","Ranaghat","Haldia","Raiganj","Krishnanagar","Nabadwip","Medinipur","Jalpaiguri","Balurghat","Basirhat","Bankura","Chakdaha","Darjeeling","Alipurduar","Purulia","Jangipur","Bangaon","Cooch Behar", "Bankura", "Garhbeta", "Kakdwip", "Berhampore", "Bolpur", "Kolkata","Asansol","Bardhaman","Khardaha","Rajarhat","Sonarpur", "Kakdwip", "Namkhana", "Contai", "KashiNagar", "Rajpur", "Habra", "Bangaon", "Kharagpur", "Ranaghat", "Nabadwip", 
"Basirhat", "Katwa", "Ghatal")

#brands <- c("lumia", "iphone", "galaxy", "aircel", "vodafone", "airtel", "mts")
brands <- c("lumia", "iphone", "galaxy")