rm(list = ls())
library(readr)
library(text2vec)
library(data.table)
library(magrittr)
library(RCurl)
library(XML)
library(pROC)
library(glmnet)
library(AUC)


#start.time = proc.time()
#1. read data 
all = read.table("data.tsv", stringsAsFactors = F, header = T)
splits = read.table("splits.csv", header = T)
s = 3
Myvocab = read.table("myVocab.txt")

# 1. Remove uselesss variables
all$review = gsub('<.*?>', ' ', all$review)

# Split Train and Test
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]



stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")


# Create a vocabulary-based DTM
prep_fun = tolower
tok_fun = word_tokenizer
it_train = itoken(train$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$new_id, 
                  progressbar = FALSE)

vocab = create_vocabulary(it_train,ngram=c(1L, 2L), stopwords = stop_words)
vectorizer = vocab_vectorizer(vocab)

# create dtm_train with new pruned vocabulary vectorizer
dtm_train  = create_dtm(it_train, vectorizer)


it_test = test$review %>% prep_fun %>% tok_fun %>% 
  # turn off progressbar because it won't look nice in rmd
  itoken(ids = test$id, progressbar = FALSE)
dtm_test = create_dtm(it_test, vectorizer)


set.seed(500)
NFOLDS = 10

train_X = dtm_train[,which(colnames(dtm_train)%in%Myvocab[,1])]
test_X = dtm_test[,which(colnames(dtm_test)%in%Myvocab[,1])]

mycv = cv.glmnet(x=train_X, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=train_X, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)


logit_pred = predict(myfit,test_X, type = "response")

#glmnet:::auc(test$sentiment, logit_pred)


# Results
results = data.frame(new_id = test$new_id, prob = logit_pred)
colnames(results) = c("new_id", "prob")
write.table(results,"mysubmission.txt", sep = ",", col.names= T, row.names = F, quote=F)
#running_time = (proc.time() - start.time)[1]
#write.table(results,file = paste("Result_3",".txt", sep = ""), sep = ",", col.names= T, row.names = F, quote=F)


