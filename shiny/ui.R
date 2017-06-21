require(rCharts)
options(RCHART_LIB = 'polycharts')

shinyUI(pageWithSidebar(
	
	headerPanel("Comparing Smartphone Brands in India with Sentiment Analysis on the related tweets (Mar 17th - April 4th 2014) with R/Shiny"),
    
	sidebarPanel(
		conditionalPanel(condition="input.conditionedPanels == 'Map'",  
		                 tags$div(title="Click here to select brand",
		                          radioButtons("rbrand1", "Brands", c(brands), selected = brands[1]))
		),
		conditionalPanel(condition="input.conditionedPanels == 'Trend' | input.conditionedPanels == 'Polarity'", 
			tags$div(title="Click here to slide through days",
			selectInput(inputId = "day",
			  label = "Select day to compare daily #tweets and overall sentiment scores",
			  choices = sort(unique(twits_trend$day)),
			  selected = "2014-03-18"))
		),
		conditionalPanel(condition="input.conditionedPanels == 'Polarity' | input.conditionedPanels == 'Pie'", 
			tags$div(title="Click here to slide through brands",
			radioButtons("rbrand", "Brands", c(brands), selected = brands[1]))
		),
		conditionalPanel(condition="input.conditionedPanels == 'Trend' | input.conditionedPanels == 'Compare Brands'", 
        	checkboxGroupInput("cbrand", "Brands", c(brands), selected = brands[1:3])
		),
		conditionalPanel(condition="input.conditionedPanels == 'Topic Clusters'", 
			selectInput(inputId = "ncluster",
			  label = "Select Number of Clusters",
			  choices = 2:19,
			  selected = 3)
		)
    ),

	mainPanel(
		tabsetPanel(
			tabPanel("Map", plotOutput("plotMap")), 
			tabPanel("Trend", showOutput("plotTweet", "polycharts"), showOutput("plotSentiment", "polycharts")), 
			tabPanel("Polarity", showOutput("plotPolarity", "polycharts")), 
			tabPanel("Pie", plotOutput("plotPie")), 
			tabPanel("Wordcloud", plotOutput("plotWord")), 
			tabPanel("Compare Brands", plotOutput("plotCompare")), 
			tabPanel("Topic Clusters", plotOutput("plotClust"), plotOutput("plotCClust")),
			tabPanel("Tweets", dataTableOutput("tweets")), 
			id = "conditionedPanels" 
		)
    )
  
))
