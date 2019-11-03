rm(list = ls())
library(dplyr)
library(text2vec)
library(tidytext)


all = read.table("data.tsv",stringsAsFactors = F,header = T)
# Remove HTML tags
#all$review <- gsub('<.*?>', ' ', all$review)
# Remove grammar/punctuation
#all$review <- tolower(gsub('[[:punct:]]', '', all$review))
# Remove numbers
#all$review <- gsub('[[:digit:]]+', '', all$review)

stop_words = c("i", "me", "my", "myself",
               "we", "our", "ours", "ourselves",
               "you", "your", "yours",
               "their", "they", "his", "her",
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were",
               "him", "himself", "has", "have",
               "it", "its", "of", "one", "for",
               "the", "us", "this")

# Train-test split
splits = read.table("splits.csv", header = TRUE) 
s = 3
test = all[which(all$new_id %in% splits[,s]),]
train = all[-which(all$new_id %in% splits[,s]),]


write.csv(train,"traindatasplit3.csv", row.names = FALSE)
write.csv(test,"testdatasplit3.csv", row.names = FALSE)



# Create a vocabulary-based DTM
prep_fun = tolower
tok_fun = word_tokenizer
train_tokens = train$review %>% prep_fun %>% tok_fun
it_train = itoken(train_tokens, ids = train$id, progressbar = FALSE)

#Pruning vocabulary
vocab = create_vocabulary(it_train, ngram = c(1L, 2L),
                          stopwords = stop_words)
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 3,
                                doc_proportion_max = 1,
                                doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, vectorizer)               

test_tokens = test$review %>% prep_fun %>% tok_fun
it_test = itoken(test_tokens, ids = test$id, progressbar = FALSE)
dtm_test = create_dtm(it_test, vectorizer)




v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = apply(dtm_train[ytrain==1, ], 2, mean)
summ[,2] = apply(dtm_train[ytrain==1, ], 2, var)
summ[,3] = apply(dtm_train[ytrain==0, ], 2, mean)
summ[,4] = apply(dtm_train[ytrain==0, ], 2, var)
n1=sum(ytrain)
n=length(ytrain)
n0= n - n1
myp = (summ[,1] - summ[,3])/sqrt(summ[,2]/n1 + summ[,4]/n0)

words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:3000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]


tmp = words[id]
tmp.term =  gsub(" ","_", tmp)

write.table(tmp.term,"myvocab.txt",row.names = FALSE, sep = ",", col.names = FALSE, quote = FALSE)
write.table(pos.list,"positive.txt",row.names = FALSE, sep = ",", col.names = FALSE, quote = FALSE)
write.table(neg.list,"negative.txt",row.names = FALSE, sep = ",", col.names = FALSE, quote = FALSE)


Myvocab = read.table("myvocab.txt")

library(glmnet)
set.seed(500)
NFOLDS = 10
train_x = dtm_train[,which(colnames(dtm_train)%in%Myvocab [,1])]
test_x = dtm_test[,which(colnames(dtm_test)%in%Myvocab[,1])]
#dtm_train[, id]

mycv = cv.glmnet(x=train_x, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)   #ridge
myfit = glmnet(x=train_x, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0) 

logit_pred = predict(myfit, test_x, type = "response")
glmnet:::auc(test$sentiment, logit_pred) 

