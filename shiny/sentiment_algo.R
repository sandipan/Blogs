preprocess.text <- function(sentence) {

	sentence = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", sentence)
	sentence = gsub("@\\w+", "", sentence)
	sentence = gsub("[[:punct:]]", "", sentence)
	sentence = gsub("[[:digit:]]", "", sentence)
	sentence = gsub("http\\w+", "", sentence)
	sentence = gsub("[ \t]{2,}", " ", sentence)
	sentence = gsub("^\\s+|\\s+$", " ", sentence)
	
	return(sentence)
}

getCorpus <- function(docs) {
	corpus <- Corpus(VectorSource(docs))
	corpus <- tm_map(corpus, content_transformer(tolower))
	corpus <- tm_map(corpus, removePunctuation)
	corpus <- tm_map(corpus, function(x)removeWords(x,stopwords()))
	return(corpus)
}

getHierarchical <- function(corpus) {
	dtm <- TermDocumentMatrix(corpus)
	spdtm <- removeSparseTerms(dtm, sparse=0.95)
	d <- as.data.frame(inspect(spdtm))
	#nrow(d)
	#ncol(d)
	d.scale <- scale(d)
	d <- dist(d.scale, method = "euclidean")
	fit <- hclust(d, method="ward.D")
	return(fit)
}

sentiment.score.simple <- function(sentences, pos.words, neg.words, companyName, .progress='none') {  
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    sentence <- gsub('[[:punct:]]', '', sentence)
    sentence <- gsub('[[:cntrl:]]', '', sentence)
    sentence <- gsub('\\d+', '', sentence)
    
    sentence = try(u_to_lower_case(sentence), TRUE)
    word.list = str_split(sentence, '\\s+')
    words = unlist(word.list)
    
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences, name=companyName)
  return(scores.df)
}

score.sentiment <- function(sentences, pos.words, neg.words, .progress='none')
{
	scores = laply(sentences, function(sentence, pos.words, neg.words) {
		
		sentence = gsub('[[:punct:]]', '', sentence)
		sentence = gsub('[[:cntrl:]]', '', sentence)
		sentence = gsub('\\d+', '', sentence)
		sentence = tolower(sentence)

		word.list = str_split(sentence, '\\s+')
		words = unlist(word.list)

		pos.matches = match(words, pos.words)
		neg.matches = match(words, neg.words)
	
		pos.matches = !is.na(pos.matches)
		neg.matches = !is.na(neg.matches)

		score = sum(pos.matches) - sum(neg.matches)

		return(score)
	}, pos.words, neg.words, .progress=.progress )

	scores.df = data.frame(score=scores, text=sentences)
	return(scores.df)
}

score.sentiment.polarity <- function(sentences, .progress='none') {

	scores = laply(sentences, function(sentence) {
		sentence <- preprocess.text(sentence)
		try.error = function(x)
		{
		   y = NA
		   try_error = tryCatch(tolower(x), error=function(e) e)
		   if (!inherits(try_error, "error"))
		   y = tolower(x)
		   return(y)
		}
		sentence = sapply(sentence, try.error)
		sentence = sentence[!is.na(sentence)]
		names(sentence) = NULL
		class_pol = classify_polarity(sentence, algorithm="bayes")
		polarity = class_pol[,4]
		return(polarity)
	}, .progress=.progress )

	scores.df = data.frame(score=scores, text=sentences)
	return(scores.df)
}

sentiment.score.multiclass <- function(sentences, vNegTerms, negTerms, posTerms, vPosTerms){
  final_scores <- matrix('', 0, 5)
  scores <- lapply(sentences, function(sentence, vNegTerms, negTerms, posTerms, vPosTerms){
    initial_sentence <- sentence
    sentence <- gsub('[[:punct:]]', '', sentence)
    sentence <- gsub('[[:cntrl:]]', '', sentence)
    sentence <- gsub('\\d+', '', sentence)
    sentence <- tolower(sentence)
    wordList <- str_split(sentence, '\\s+')
    words <- unlist(wordList)
    vPosMatches <- match(words, vPosTerms)
    posMatches <- match(words, posTerms)
    vNegMatches <- match(words, vNegTerms)
    negMatches <- match(words, negTerms)
    vPosMatches <- sum(!is.na(vPosMatches))
    posMatches <- sum(!is.na(posMatches))
    vNegMatches <- sum(!is.na(vNegMatches))
    negMatches <- sum(!is.na(negMatches))
    score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
    newrow <- c(initial_sentence, score)
    print(paste("Here::", newrow))
	final_scores <- rbind(final_scores, newrow)
    return(final_scores)
  }, vNegTerms, negTerms, posTerms, vPosTerms)
  return(scores)
}

pos <- readLines("positive_words.txt")
neg <- readLines("negative_words.txt")
