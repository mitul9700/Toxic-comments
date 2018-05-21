#### TOXIC COMMENTS CLASSIFICATION 
### AUTHOR : MITUL SOLANKI


setwd("/Users/mitulsolanki/edwisor_proj/Toxic")
getwd()
rm(list=ls())


library(e1071)
library(tidyverse)
library(tidytext)
library(stringr)
library(wordcloud)
library(tm)
library(SnowballC)
library(caret)
library(ggplot2)

# loading training , test , sample submission dataset
train = read_csv("train.csv")
test = read_csv("test.csv")
submission = read_csv("sample_submission.csv")

# formatting the comment variable of train dataset
train$comment_text=gsub("'|\"|'|???|???|\"|\n|,|\\.|???|\\?|\\+|\\-|\\/|\\=|\\(|\\)|???", "", 
                        train$comment_text)

# adding new variable to check the length of each comment in both dataset
train$comment_length = str_count(train$comment_text)
test$comment_length = str_count(test$comment_text)

# visualizing the words distribution
ggplot(data = train, aes(x =comment_length)) + 
  geom_histogram(fill = 'orange', bins = 50) +
  labs(title = 'Words distribution')

# extracting the words from comments from train dataset
words_train = train %>%
  unnest_tokens(word, comment_text) %>%
  count(toxic,severe_toxic,obscene,threat,insult,identity_hate,word) %>%
  ungroup()

# finding the total number of words in each class
total_words = words_train %>% 
  group_by(toxic,severe_toxic,obscene,threat,insult,identity_hate) %>% 
  summarize(total = sum(n))

# there are 41 observations in total_words so we can label them as categories
category =1:41
total_words$category = category
words_train = left_join(words_train, total_words) #applying left join on words in train and total words

# binding the tf and idf
words_train = words_train %>%
  bind_tf_idf(word, category, n)

# arranging by if_idf and levels will be the reverse order of unique words
plot_train_words = words_train %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word))))

# building the wordcloud of 100 words from above plot_train_words by tf_idf
plot_train_words %>%
  with(wordcloud(word, tf_idf, max.words = 100,colors=brewer.pal(8, "Dark2")))

# plotting top 40 words according to tf-idf
plot_train_words %>% 
  top_n(40) %>%
  ggplot(aes(word, tf_idf)) +
  geom_col(fill = 'orange') +
  labs(x = NULL, y = "tf-idf") +
  coord_flip() +
  theme_bw()

# plotting the tf-idf for toxic category top 15 words
plot_train_words %>%
  filter(toxic == 1 ) %>%
  top_n(15) %>%
  ggplot(aes(word, tf_idf)) +
  geom_col(fill = 'blue') +
  labs(x = NULL, y = "tf-idf") +
  coord_flip() +
  theme_bw()

# likewise we can plot tf-idf for all categories just by replacing the ""category name=1"
# example toxic =1 , severe_toxic =1 , threat =1 etc

# replacing the alpha-numerics from comments with space and converting it into ascii format
train$comment_text = iconv(train$comment_text, 'UTF-8', 'ASCII')
train$comment_text=str_replace_all(train$comment_text,"[^[:alnum:]]", " ")

# building corpus and performing text processing 
corpus = VCorpus(VectorSource(train$comment_text))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# building document_term_matrix from the corpus , removing the sparse terms and converting it into data frame
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.99)
dataset = as.data.frame(as.matrix(dtm))
dataset$toxic = NULL
dataset$severe_toxic = NULL
dataset$obscene = NULL
dataset$threat = NULL
dataset$insult = NULL
dataset$identity_hate = NULL

####### above steps repeated on test dataset
test$comment_text = iconv(test$comment_text, 'UTF-8', 'ASCII')
test$comment_text=str_replace_all(test$comment_text,"[^[:alnum:]]", " ")

corpus = VCorpus(VectorSource(test$comment_text))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)


dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.99)
dataset_test = as.data.frame(as.matrix(dtm))

# making same column names for test and train dataset
same_col_names = intersect(colnames(dataset),colnames(dataset_test))

# splitting data into train and test
dataset = dataset[ , (colnames(dataset) %in% same_col_names)]
dataset_test = dataset_test[ , (colnames(dataset_test) %in% same_col_names)]

# fitting the naive-bayes model on training set which will yield the probabilities

# calculating for "toxic"
new_dataset = dataset
new_dataset$toxic = train$toxic
new_dataset$toxic = as.factor(new_dataset$toxic)
levels(new_dataset$toxic) = make.names(unique(new_dataset$toxic))

model = naiveBayes(toxic ~ . , data = new_dataset)
predict_toxic = predict(model , newdata = dataset_test , type = 'raw')
predict_toxic = as.data.frame(predict_toxic)
submission$toxic = predict_toxic$X1

### severe_toxic
new_dataset = dataset
new_dataset$severe_toxic = train$severe_toxic
new_dataset$severe_toxic = as.factor(new_dataset$severe_toxic)
levels(new_dataset$severe_toxic) = make.names(unique(new_dataset$severe_toxic))

model = naiveBayes(severe_toxic ~ . , data = new_dataset)
predict_severe_toxic = predict(model , newdata = dataset_test , type = 'raw')
predict_severe_toxic = as.data.frame(predict_severe_toxic)
submission$severe_toxic = predict_severe_toxic$X1

######### obscene

new_dataset = dataset
new_dataset$obscene = train$obscene
new_dataset$obscene = as.factor(new_dataset$obscene)
levels(new_dataset$obscene) = make.names(unique(new_dataset$obscene))

model = naiveBayes(obscene ~ . , data = new_dataset)
predict_obscene = predict(model , newdata = dataset_test , type = 'raw')
predict_obscene = as.data.frame(predict_obscene)
submission$obscene = predict_obscene$X1

########## threat

new_dataset = dataset
new_dataset$threat = train$threat
new_dataset$threat = as.factor(new_dataset$threat)
levels(new_dataset$threat) = make.names(unique(new_dataset$threat))

model = naiveBayes(threat ~ . , data = new_dataset)
predict_threat = predict(model , newdata = dataset_test , type = 'raw')
predict_threat = as.data.frame(predict_threat)
submission$threat = predict_threat$X1

########## insult

new_dataset = dataset
new_dataset$insult = train$insult
new_dataset$insult = as.factor(new_dataset$insult)
levels(new_dataset$insult) = make.names(unique(new_dataset$insult))

model = naiveBayes(insult ~ . , data = new_dataset)
predict_insult = predict(model , newdata = dataset_test , type = 'raw')
predict_insult = as.data.frame(predict_insult)
submission$insult = predict_insult$X1

######## identity_hate

new_dataset = dataset
new_dataset$identity_hate = train$identity_hate
new_dataset$identity_hate = as.factor(new_dataset$identity_hate)
levels(new_dataset$identity_hate) = make.names(unique(new_dataset$identity_hate))

model = naiveBayes(identity_hate ~ . , data = new_dataset)
predict_identity_hate = predict(model , newdata = dataset_test , type = 'raw')
predict_identity_hate = as.data.frame(predict_identity_hate)
submission$identity_hate = predict_identity_hate$X1

write.csv(submission, 'Comments_classification_naive_bayes.csv', row.names = F)

### but by looking the prediction file of naive bayes it doesn't look much acurate as many variables
### has 1 as a probabilty for same rows.

### so we will try to classify using XGboost modeling.

## For toxic

new_dataset = dataset
new_dataset$toxic = train$toxic
new_dataset$toxic = as.factor(new_dataset$toxic)
levels(new_dataset$toxic) = make.names(unique(new_dataset$toxic))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                       max_depth = 3,
                       eta = .05,
                       gamma = 0,
                       colsample_bytree = .8,
                       min_child_weight = 1,
                       subsample = 1)


set.seed(13)

Toxic_XGB = train(toxic ~ ., data = new_dataset,
                 method = "xgbTree",trControl = fitControl,
                 tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_toxic = predict(Toxic_XGB,dataset_test,type = 'prob')
predict_toxic = as.data.frame(predict_toxic)
submission$toxic = predict_toxic$X1

## For severe_toxic

new_dataset = dataset
new_dataset$severe_toxic = train$severe_toxic
new_dataset$severe_toxic = as.factor(new_dataset$severe_toxic)
levels(new_dataset$severe_toxic) = make.names(unique(new_dataset$severe_toxic))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                      max_depth = 3,
                      eta = .05,
                      gamma = 0,
                      colsample_bytree = .8,
                      min_child_weight = 1,
                      subsample = 1)


set.seed(13)

Severe_Toxic_XGB = train(severe_toxic ~ ., data = new_dataset,
                  method = "xgbTree",trControl = fitControl,
                  tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_severe_toxic = predict(Severe_Toxic_XGB,dataset_test,type = 'prob')
predict_severe_toxic = as.data.frame(predict_toxic)
submission$severe_toxic = predict_severe_toxic$X1

## For Obscene

new_dataset = dataset
new_dataset$obscene = train$obscene
new_dataset$obscene = as.factor(new_dataset$obscene)
levels(new_dataset$obscene) = make.names(unique(new_dataset$obscene))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                      max_depth = 3,
                      eta = .05,
                      gamma = 0,
                      colsample_bytree = .8,
                      min_child_weight = 1,
                      subsample = 1)


set.seed(13)

Obscene_XGB = train(obscene ~ ., data = new_dataset,
                         method = "xgbTree",trControl = fitControl,
                         tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_obscene = predict(Obscene_XGB,dataset_test,type = 'prob')
predict_obscene = as.data.frame(predict_toxic)
submission$obscene = predict_obscene$X1

## For Threat

new_dataset = dataset
new_dataset$threat = train$threat
new_dataset$threat = as.factor(new_dataset$threat)
levels(new_dataset$threat) = make.names(unique(new_dataset$threat))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                      max_depth = 3,
                      eta = .05,
                      gamma = 0,
                      colsample_bytree = .8,
                      min_child_weight = 1,
                      subsample = 1)


set.seed(13)

Threat_XGB = train(threat ~ ., data = new_dataset,
                    method = "xgbTree",trControl = fitControl,
                    tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_threat = predict(Threat_XGB,dataset_test,type = 'prob')
predict_threat = as.data.frame(predict_threat)
submission$threat = predict_threat$X1

## For Insult

new_dataset = dataset
new_dataset$insult = train$insult
new_dataset$insult = as.factor(new_dataset$insult)
levels(new_dataset$insult) = make.names(unique(new_dataset$insult))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                      max_depth = 3,
                      eta = .05,
                      gamma = 0,
                      colsample_bytree = .8,
                      min_child_weight = 1,
                      subsample = 1)


set.seed(13)

Insult_XGB = train(insult ~ ., data = new_dataset,
                   method = "xgbTree",trControl = fitControl,
                   tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_insult = predict(Insult_XGB,dataset_test,type = 'prob')
predict_insult = as.data.frame(predict_insult)
submission$insult = predict_threat$X1

## For Identity_hate

new_dataset = dataset
new_dataset$identity_hate = train$identity_hate
new_dataset$identity_hate = as.factor(new_dataset$identity_hate)
levels(new_dataset$identity_hate) = make.names(unique(new_dataset$identity_hate))

fitControl = trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)

xgbGrid = expand.grid(nrounds = 500,
                      max_depth = 3,
                      eta = .05,
                      gamma = 0,
                      colsample_bytree = .8,
                      min_child_weight = 1,
                      subsample = 1)


set.seed(13)

Identity_hate_XGB = train(identity_hate ~ ., data = new_dataset,
                   method = "xgbTree",trControl = fitControl,
                   tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", maximize=FALSE)

predict_identity_hate = predict(Identity_hate_XGB,dataset_test,type = 'prob')
predict_identity_hate = as.data.frame(predict_identity_hate)
submission$identity_hate = predict_identity_hate$X1

write.csv(submission, 'Comments_classification_XGBoost.csv', row.names = F)
