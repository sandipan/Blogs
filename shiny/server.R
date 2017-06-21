require(rCharts)
options(RCHART_WIDTH = 800)

shinyServer(function(input, output) {
	output$plotWord <- renderImage({
	  return(list(
	    src = "wordcloud.png",
	    contentType = "image/png",
	    width = 800,
	    height = 600,
	    alt = "Face"
	  ))
	}, deleteFile = FALSE)
	output$plotClust <- renderPlot({
		print(as.integer(input$ncluster))
		plotHierarchical(fit, as.integer(input$ncluster))
	})
	output$plotCClust <- renderPlot({
		print(as.integer(input$ncluster))
		plotHierarchicalCol(fit, as.integer(input$ncluster))
	})
  output$plotMap <- renderImage({
		device = input$rbrand1
		return(list(
		  src = mapImages[[device]],
		  contentType = "image/png",
		  width = 1200,
		  height = 1000,
		  alt = "Face"
		))
  }, deleteFile = FALSE)
  output$plotCompare <- renderPlot({
		plotComparison(twits_trend, "Smartphone", input$cbrand)
	})
	output$plotTweet <- renderChart({
		twits <- subset(twits_trend, day == input$day & name == input$cbrand)
		twits$time <- as.character(twits$time)
		twits <- sqldf("select name, hour, count(score) as score from twits group by hour, name")
		p1 <- rPlot(score ~ hour, color = 'name', type = 'line', data = twits)
		p1$guides(y = list(min = 0, max = max(twits$score) + 10, title = ""))
		p1$guides(y = list(title = "#Tweets"))
		p1$addParams(height = 250, dom = 'plotTweet')
		return(p1)
  })
  output$plotSentiment <- renderChart({
		twits <- subset(twits_trend, day == input$day & name == input$cbrand)
		twits$time <- as.character(twits$time)
		twits <- sqldf("select name, hour, sum(score) as score from twits group by hour, name")
		p2 <- rPlot(score ~ hour, color = 'name', type = 'line', data = twits)
		p2$guides(y = list(min = min(twits$score) - 10, max = max(twits$score) + 10, title = ""))
		p2$guides(y = list(title = "Sentiment Scores"))
		p2$addParams(height = 250, dom = 'plotSentiment')
		return(p2)
  })
  output$plotPolarity <- renderChart({
		twits <- subset(twits_trend, day == input$day & name == input$rbrand)
		twits$time <- as.character(twits$time)
		twits$polarity <- 'neutral'
		twits[twits$score > 0,]$polarity <- 'positive'
		twits[twits$score < 0,]$polarity <- 'negative'
		twits <- sqldf("select polarity, hour, count(polarity) as score from twits group by hour, polarity")
		p3 <- rPlot(score ~ hour, color = 'polarity', type = 'line', data = twits)
		p3$guides(y = list(min = 0, max = max(twits$score) + 10, title = ""))
		p3$guides(y = list(title = paste("Sentiment Polarity for", input$rbrand)))
		p3$addParams(height = 250, dom = 'plotPolarity')
		return(p3)
  })
	output$plotPie <- renderPlot({
	  twits <- subset(twits_trend, name == input$rbrand)
	  twits$polarity <- 'neutral'
	  twits[twits$score > 0,]$polarity <- 'positive'
	  twits[twits$score < 0,]$polarity <- 'negative'
	  p4 <- ggplot(twits, aes(x = factor(1), fill = factor(polarity))) +
	    geom_bar(width = 1)+ coord_polar(theta = "y")
	  return(p4)
	})
  output$tweets <- renderDataTable({
		data.frame(twits_trend)
	}, options = list(bSortClasses = TRUE, aLengthMenu = c(5, 30, 50), iDisplayLength = 5))
	output$mpgPlot <- renderPlot({
		boxplot(score~name, data=twits)
	})
})
