###
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
	require(grid)

	# Make a list from the ... arguments and plotlist
	plots <- c(list(...), plotlist)
	numPlots = length(plots)

	# If layout is NULL, then use 'cols' to determine layout
	if (is.null(layout)) {
		# Make the panel
		# ncol: Number of columns of plots
		# nrow: Number of rows needed, calculated from # of cols
		layout <- matrix(seq(1, cols * ceiling(numPlots/cols)), ncol = cols, nrow = ceiling(numPlots/cols))
	}

	if (numPlots==1) {
		print(plots[[1]])
	} else {
		# Set up the page
		grid.newpage()
		pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

		# Make each plot, in the correct location
		for (i in 1:numPlots) {
			# Get the i,j matrix positions of the regions that contain this subplot
			matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
			print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row, layout.pos.col = matchidx$col))
		}
	}
}

###
plotComparison <- function(text_score, text_type, brand.names = unique(text_score$name)) {

	text_score <- subset(text_score, name == brand.names)
	
	text_score$very.pos.bool = text_score$score >= 1
	text_score$very.pos = as.numeric(text_score$very.pos.bool)
	
	text_score$very.neg.bool = text_score$score <= -1
	text_score$very.neg = as.numeric(text_score$very.neg.bool)
	 
	score.df = ddply(text_score, c('name'), summarise, very.pos.count=sum(very.pos), very.neg.count=sum(very.neg))
	 
	score.df$very.total = score.df$very.pos.count +  score.df$very.neg.count
	score.df$percentScore = round(100 * score.df$very.pos.count / score.df$very.total)
	score.df$percentLooser = round(100 * score.df$very.neg.count / score.df$very.total)
	
	#print(head(text_score))
	 
	g <- ggplot(data=text_score, mapping=aes(x=score, fill=name))
	g <- g + geom_histogram(binwidth=1)
	g <- g + facet_grid(name~.)
	g <- g + theme_bw() + scale_fill_brewer(type="qual", palette="Set1")
	 
	happy <- ggplot(score.df, aes(x=name, y=percentScore), fill='orange') + 
	  geom_bar(stat='identity', position='dodge') +
	  ylab("Relative Happiness") + xlab(paste(text_type, "Ranking")) + theme_bw()
	
	unhappy <- ggplot(score.df, aes(x=name, y=percentLooser), fill='brown') + 
	  geom_bar(stat='identity', position='dodge') +
	  ylab("Relative Unhappiness") + xlab(paste(text_type, "Ranking")) + theme_bw()
	
	sco <-  ggplot(score.df, aes(x=name, y=percentLooser), fill='brown') + 
	  geom_bar(stat='identity', position='dodge') +
	  ylab("Social Mentions") + xlab(paste("By", text_type)) + theme_bw() 
	 
	multiplot(g, sco, happy, unhappy, cols=2)
}

plotWordCloud <- function(corpus) {
	pal2 <- brewer.pal(8,"Dark2")
	wordcloud(corpus,min.freq=2,max.words=500, random.order=T, colors=pal2)
}

getFreqItems <- function(corpus, k) {
	dtm <- TermDocumentMatrix(corpus)
	trms <- findFreqTerms(dtm, lowfreq=k)	
	#findAssocs(twit.dtm, trms[1], 0.20)
	return(trms)
}

plotHierarchical <- function(fit, n) {
	plot(fit, sub = "")
	#plot(fit, hang=-1)
	groups <- cutree(fit, k=n)
	rect.hclust(fit, k=n, border="red")
}

plotHierarchicalCol <- function(fit, n) {
	op = par(bg = "gray15")
	cols = hsv(runif(n), 1, 1, 0.8)
	A2Rplot(fit, k = n, boxes = TRUE, col.up = "gray50", col.down = cols) #rainbow(n))
}

plotHierarchicalgg <- function(fit)	{
	ggdendrogram(fit, theme_dendro = FALSE)
}

plotMap <- function() {
	
}