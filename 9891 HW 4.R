rm(list = ls())
cat("\014")

reticulate::use_python("C:/Users/eriks/anaconda3/python.exe")
library(parallel)
library(reticulate)
library(keras)
library(ggplot2)
library(glmnet)
library(pROC)
library(gridExtra)
library(randomForest)

p = 2500
reviews <- dataset_imdb(num_words = p)

word_index                   =     dataset_imdb_word_index() 
reverse_word_index           =     names(word_index) 
names(reverse_word_index)    =     word_index

x.train <- reviews$train$x
y.train <- as.numeric(reviews$train$y)

x.test <- reviews$test$x
y.test <- as.numeric(reviews$test$y)

vectorize.sequence <- function(sequence, features){
    feature.matrix <- matrix(0, nrow=length(sequence), ncol = features)
    for(i in 1:length(sequence)){
        feature.matrix[i, sequence[[i]]] <- 1
    }
    feature.matrix
}

# Subset the data
x.train <- vectorize.sequence(x.train, p)
x.test <- vectorize.sequence(x.test, p)


train.p.index = which(y.train == 1)
train.p.index = train.p.index[1:4000]
train.n.index = which(y.train == 0)

test.p.index = which(y.test == 1)
test.p.index = test.p.index[1:4000]
test.n.index = which(y.test == 0)

train.index = c(train.p.index, train.n.index)
test.index = c(test.p.index, test.n.index)

X.train = x.train[train.index, ]
Y.train = y.train[train.index]
X.test = x.test[test.index,]
Y.test = y.test[test.index]

# y.train.factor = as.factor(Y.test)


### How long will it take to cross validate and fit a random forrest?

library(tree)

dat <- data.frame(X = X.train, Y = as.factor(Y.train))
dat.test <- data.frame(X=X.test, Y = as.factor(Y.test))

tree.start <- proc.time()

tree.fit <- tree(Y~., data = dat)
opt.tree <- cv.tree(tree.fit, FUN = prune.misclass)
opt.size = opt.tree$size[which.min(opt.tree$dev)]
pruned.tree.fit = prune.misclass(tree.fit, best = opt.size)

words.rf <- randomForest(Y~., data = dat, mtry=sqrt(p))


rf.roc <- roc(dat$Y, words.rf$votes[,2])

# Make predictions
X.test.df <- data.frame(X=X.test)
test.preds <- predict(words.rf, newdata = X.test.df, type="vote")

test.rf.roc <-roc(dat.test$Y, test.preds[,2])
plot(test.rf.roc)
auc(test.rf.roc)

data(iris)


ind = sample(2, nrow(iris), replace = TRUE, prob=c(0.8, 0.2))



################
## Q1 - Lasso ##
################

lasso.alpha = 1
ridge.alpha = 0
elnet.alpha = .5


# cross validate the lambdas


start = proc.time()
lasso.cv <- cv.glmnet(X.train, Y.train,
                      family="binomial",
                      alpha = lasso.alpha,
                      type.measure = "auc")
end = proc.time()
lasso.cv.time <- end[3]-start[3]
lasso.cv.time  # 7752.98 (run 1) 1.48 hours (run 3 with ryzen 7) (run 4 - with parallel running)


#########################
start.ridge = proc.time()
ridge.cv <- cv.glmnet(X.train, Y.train,
                      family="binomial",
                      alpha = ridge.alpha,
                      type.measure = "auc")
end.ridge = proc.time()
ridge.cv.time <- end.ridge[3]-start.ridge[3]
ridge.cv.time #  839.8 (run 1) (.1999 hours run 3 with ryzen 7)

##########################

start.elnet = proc.time()
elnet.cv <- cv.glmnet(X.train, Y.train,
                      family="binomial",
                      alpha = elnet.alpha,
                      type.measure = "auc")
end.elnet = proc.time()
elnet.cv.time <- end[3]-start[3]
elnet.cv.time  #7752.98 (run 1)

total.time = (lasso.cv.time + ridge.cv.time + elnet.cv.time)
total.hours = total.time/3600
total.hours

lambda.lasso <-  lasso.cv$lambda.min # 0.00457543 (run 1)
lambda.ridge <-  ridge.cv$lambda.min # 0.1540582 (run 1)
lambda.elnet <-  elnet.cv$lambda.min # 0.00457543 (run 1)


lambda.lasso
lambda.ridge
lambda.elnet

# Train the models

lasso.train  <- glmnet(X.train, Y.train, family = "binomial",
                       alpha = lasso.alpha, lambda = lambda.lasso)

ridge.train  <-  glmnet(X.train, Y.train, family = "binomial",
                        alpha = ridge.alpha, lambda = lambda.ridge)

elnet.train  <-  glmnet(X.train, Y.train, family = "binomial",
                        alpha = elnet.alpha, lambda = lambda.elnet)


lasso.beta_    <- as.vector(lasso.train$beta)
lasso.beta0_   <- lasso.train$a0



ridge.beta_    <- as.vector(ridge.train$beta)
ridge.beta0_   <- ridge.train$a0

elnet.beta_    <- as.vector(elnet.train$beta)
elnet.beta0_   <- elnet.train$a0


xtb.lasso.train      <- X.train%*%lasso.beta_ + lasso.beta0_
xtb.ridge.train      <- X.train%*%ridge.beta_ + ridge.beta0_
xtb.elnet.train      <- X.train%*%elnet.beta_ + elnet.beta0_

xtb.lasso.test      <- X.test%*%lasso.beta_ + lasso.beta0_
xtb.ridge.test      <- X.test%*%ridge.beta_ + ridge.beta0_
xtb.elnet.test      <- X.test%*%elnet.beta_ + elnet.beta0_




## Part A & B - Top 5 Words

obh.lasso <- order(lasso.beta_)
obh.ridge <- order(ridge.beta_)
obh.elnet <- order(elnet.beta_)


max.words = 5

lasso.index.negatives = obh.lasso[1:max.words]
lasso.index.positives = obh.lasso[(p-max.words+1):p]

ridge.index.negatives = obh.ridge[1:max.words]
ridge.index.positives = obh.ridge[(p-max.words+1):p]

elnet.index.negatives = obh.elnet[1:max.words]
elnet.index.positives = obh.elnet[(p-max.words+1):p]

# cat(negative.words) # ugghhh ravenously paris's 'always faulted
# cat(positive.words) # 7 refreshing wonderfully captures noir


## Lasso Positive/Negative Words ##

lasso.negative.words = sapply(lasso.index.negatives, function(index){
    word <- if(index>=3) reverse_word_index[[(index-3)]]
    if (!is.null(word)) word else "?"    
})
lasso.positive.words = sapply(lasso.index.positives, function(index){
    word <- if(index>=3) reverse_word_index[[as.character(index-3)]]
    if (!is.null(word)) word else "?"    
})


## Ridge Positive/Negative Words ##

ridge.negative.words = sapply(ridge.index.negatives, function(index){
  word <- if(index>=3) reverse_word_index[[(index-3)]]
  if (!is.null(word)) word else "?"    
})
ridge.positive.words = sapply(ridge.index.positives, function(index){
  word <- if(index>=3) reverse_word_index[[as.character(index-3)]]
  if (!is.null(word)) word else "?"    
})

ridge.positives <- cat(ridge.positive.words)
ridge.negatives <- cat(ridge.negative.words)

## Elastic Net Positive/Negative Words ##

elnet.negative.words = sapply(elnet.index.negatives, function(index){
  word <- if(index>=3) reverse_word_index[[(index-3)]]
  if (!is.null(word)) word else "?"    
})

elnet.positive.words = sapply(elnet.index.positives, function(index){
  word <- if(index>=3) reverse_word_index[[as.character(index-3)]]
  if (!is.null(word)) word else "?"    
})

words.df <- data.frame(Method = c(rep("Lasso",5),
                                  rep("Ridge", 5),
                                  rep("Elnet",5)), 
                       Positive.Words = c(lasso.positive.words,
                                          ridge.positive.words,
                                          elnet.positive.words),
                       Negative.Words = c(lasso.negative.words,
                                          ridge.negative.words,
                                          elnet.negative.words))

words.df




Top5Words <- data.frame("Lasso.Positive"  = lasso.positive.words,
                             "Ridge.Positive"  = ridge.positive.words, 
                             "Elnet.Positive"  = elnet.positive.words,
                             "Lasso.Negative"  = lasso.negative.words,
                             "Ridge.Negative"  = ridge.negative.words, 
                             "Elnet.Negative"  = elnet.negative.words)
Top5Words

# Part 3

thresh = seq(0, 1, by=.01)
thresh.seq = seq(0, length.out = length(thresh))


p.train = sum(Y.train == 1)
n.train = sum(Y.train == 0)

p.test = sum(Y.test == 1)
n.test = sum(Y.test == 0)


# calculate observation probabilities

lasso.probs.train = exp(xtb.lasso.train)/(1+exp(xtb.lasso.train))
lasso.probs.test = exp(xtb.lasso.test)/(1+exp(xtb.lasso.test))

ridge.probs.train = exp(xtb.ridge.train)/(1+exp(xtb.ridge.train))
ridge.probs.test = exp(xtb.ridge.test)/(1+exp(xtb.ridge.test))

elnet.probs.train = exp(xtb.elnet.train)/(1+exp(xtb.elnet.train))
elnet.probs.test = exp(xtb.elnet.test)/(1+exp(xtb.elnet.test))


# Data frames to store the results of our loop below
roc.lasso = data.frame(Threshold = thresh, 
                    FPR_Train = thresh.seq,
                    TPR_Train = thresh.seq,
                    FPR_Test  = thresh.seq, 
                    TPR_Test  = thresh.seq)

roc.ridge = data.frame(Threshold = thresh, 
                       FPR_Train = thresh.seq,
                       TPR_Train = thresh.seq,
                       FPR_Test  = thresh.seq, 
                       TPR_Test  = thresh.seq)

roc.elnet = data.frame(Threshold = thresh, 
                       FPR_Train = thresh.seq,
                       TPR_Train = thresh.seq,
                       FPR_Test  = thresh.seq, 
                       TPR_Test  = thresh.seq)



for(i in 1:length(thresh)){
  
  thr = thresh[i]
  
  # Predictions based on threshold value
  y.hat.train.lasso <- ifelse(lasso.probs.train > thr, 1, 0)
  y.hat.train.ridge <- ifelse(ridge.probs.train > thr, 1, 0)
  y.hat.train.elnet <- ifelse(elnet.probs.train > thr, 1, 0)
  
  y.hat.test.lasso  <- ifelse(lasso.probs.test  > thr, 1, 0)
  y.hat.test.ridge  <- ifelse(ridge.probs.test  > thr, 1, 0)
  y.hat.test.elnet  <- ifelse(elnet.probs.test  > thr, 1, 0)
  
  # Training False & True Positives
  FP.train.lasso    <- sum(Y.train[y.hat.train.lasso==1] == 0)
  FP.train.ridge    <- sum(Y.train[y.hat.train.ridge==1] == 0)
  FP.train.elnet    <- sum(Y.train[y.hat.train.elnet==1] == 0)
  
  TP.train.lasso    <- sum(y.hat.train.lasso[Y.train==1] == 1)
  TP.train.ridge    <- sum(y.hat.train.ridge[Y.train==1] == 1)
  TP.train.elnet    <- sum(y.hat.train.elnet[Y.train==1] == 1)
  
  # Test False & True Positives
  FP.test.lasso     <- sum(Y.test[y.hat.test.lasso ==1] == 0)
  FP.test.ridge     <- sum(Y.test[y.hat.test.ridge ==1] == 0)
  FP.test.elnet     <- sum(Y.test[y.hat.test.elnet ==1] == 0)

  TP.test.lasso     <- sum(y.hat.test.lasso[Y.test==1] == 1)
  TP.test.ridge     <- sum(y.hat.test.ridge[Y.test==1] == 1)
  TP.test.elnet     <- sum(y.hat.test.elnet[Y.test==1] == 1)
 
  # Store the values into their data frames
  roc.lasso[i,2] = FP.train.lasso/n.train
  roc.lasso[i,3] = TP.train.lasso/p.train
  roc.lasso[i,4] = FP.test.lasso/n.test
  roc.lasso[i,5] = TP.test.lasso/p.test
  
  roc.ridge[i,2] = FP.train.ridge/n.train
  roc.ridge[i,3] = TP.train.ridge/p.train
  roc.ridge[i,4] = FP.test.ridge/n.test
  roc.ridge[i,5] = TP.test.ridge/p.test
  
  roc.elnet[i,2] = FP.train.elnet/n.train
  roc.elnet[i,3] = TP.train.elnet/p.train
  roc.elnet[i,4] = FP.test.elnet/n.test
  roc.elnet[i,5] = TP.test.elnet/p.test
}


# Calculate the AUC for each of the methods

lasso.train.auc <- auc(roc(as.vector(lasso.probs.train), as.factor(Y.train)))
ridge.train.auc <- auc(roc(as.vector(ridge.probs.train), as.factor(Y.train)))
elnet.train.auc <- auc(roc(as.vector(elnet.probs.train), as.factor(Y.train)))

lasso.test.auc <- auc(roc(as.vector(lasso.probs.test), as.factor(Y.test)))
ridge.test.auc <- auc(roc(as.vector(ridge.probs.test), as.factor(Y.test)))
elnet.test.auc <- auc(roc(as.vector(elnet.probs.test), as.factor(Y.test)))


(elnet.train.auc - lasso.train.auc)/elnet.train.auc
elnet.test.

# Create ROC Plots for Lasso, Ridge, El-Net

# Lasso ROC Plot
lasso.roc.plot <- roc.lasso %>% ggplot() +
  geom_line(aes(x = FPR_Train,
                y = TPR_Train,
                color = "Train")) +
  geom_line(aes(x =FPR_Test,
                y = TPR_Test,
                color = "Test")) +
  labs(x = "False Positive Rate", 
       y = "True Positive Rate", 
       title = "Lasso ROC Curve",
       subtitle = "Unweighted Model",
       color = "ROC Curve") +
  annotate(geom = "text", x = .50, y = .25,
           label = paste0("AUC Train =", round(lasso.train.auc,5), sep = " ")) +
  annotate(geom = "text", x = .50, y = .20,
           label = paste0("AUC Test = ", round(lasso.test.auc,5), sep = " "))
  
# Ridge ROC Plot
ridge.roc.plot <- roc.ridge %>% ggplot() +
  geom_line(aes(x = FPR_Train,
                y = TPR_Train,
                color = "Train")) +
  geom_line(aes(x =FPR_Test,
                y = TPR_Test,
                color = "Test")) +
  labs(x = "False Positive Rate", 
       y = "True Positive Rate", 
       title = "Ridge ROC Curve",
       subtitle = "Unweighted Model",
       color = "ROC Curve") +
  annotate(geom = "text", x = .50, y = .25,
           label = paste0("AUC Train =", round(ridge.train.auc,5), sep = " ")) +
  annotate(geom = "text", x = .50, y = .20,
           label = paste0("AUC Test = ", round(ridge.test.auc, 5), sep = " "))

# Elastic Net ROC Plot
elnet.roc.plot <- roc.elnet %>% ggplot() +
  geom_line(aes(x = FPR_Train,
                y = TPR_Train,
                color = "Train")) +
  geom_line(aes(x =FPR_Test,
                y = TPR_Test,
                color = "Test")) +
  labs(x = "False Positive Rate", 
       y = "True Positive Rate", 
       title = "El-Net ROC Curve",
       subtitle = "Unweighted Model",
       color = "ROC Curve") +
  annotate(geom = "text", x = .50, y = .25,
           label = paste0("AUC Train =", round(elnet.train.auc,5), sep = " ")) +
  annotate(geom = "text", x = .50, y = .20,
           label = paste0("AUC Test = ", round(elnet.test.auc, 5), sep = " "))

ROC.plots <- grid.arrange(lasso.roc.plot, ridge.roc.plot, elnet.roc.plot, ncol = 3)


ROC.plots


## Parts 4 & 5

method = c("Lasso", "Ridge", "ElNet")


lasso.type.rates <- data.frame(Threshold = thresh,
                               Type.1.train = roc.lasso$FPR_Train,
                               Type.2.train = (1 - roc.lasso$TPR_Train),
                               Difference.train = abs((1-roc.lasso$TPR_Train)-roc.lasso$FPR_Train),
                               Type.1.test = roc.lasso$FPR_Test,
                               Type.2.test = (1 - roc.lasso$TPR_Test),
                               Difference.test = abs((1-roc.lasso$TPR_Test)-roc.lasso$FPR_Test))

ridge.type.rates <- data.frame(Threshold = thresh,
                               Type.1.train = roc.ridge$FPR_Train,
                               Type.2.train = (1 - roc.ridge$TPR_Train),
                               Difference.train = abs((1-roc.ridge$TPR_Train)-roc.ridge$FPR_Train),
                               Type.1.test = roc.ridge$FPR_Test,
                               Type.2.test = (1 - roc.ridge$TPR_Test),
                               Difference.test = abs((1-roc.ridge$TPR_Test)-roc.ridge$FPR_Test))

elnet.type.rates <- data.frame(Threshold = thresh,
                               Type.1.train = roc.elnet$FPR_Train,
                               Type.2.train = (1 - roc.elnet$TPR_Train),
                               Difference.train = abs((1-roc.elnet$TPR_Train)-roc.elnet$FPR_Train),
                               Type.1.test = roc.elnet$FPR_Test,
                               Type.2.test = (1 - roc.elnet$TPR_Test),
                               Difference.test = abs((1-roc.elnet$TPR_Test)-roc.elnet$FPR_Test))




# For theta = 0.5 what are the type 1 and 2 errors
theta5.df <- data.frame(Method = method, 
                        Threshold = c(0.5, 0.5, 0.5),
                        Type.1.Train = c(lasso.type.rates[lasso.type.rates$Threshold == 0.5, 2],
                                         ridge.type.rates[ridge.type.rates$Threshold == 0.5, 2],
                                         elnet.type.rates[elnet.type.rates$Threshold == 0.5, 2]),
                        
                        Type.2.Train = c(lasso.type.rates[lasso.type.rates$Threshold == 0.5, 3],
                                         ridge.type.rates[ridge.type.rates$Threshold == 0.5, 3],
                                         elnet.type.rates[elnet.type.rates$Threshold == 0.5, 3]),
                       
                        Type.1.Test  = c(lasso.type.rates[lasso.type.rates$Threshold == 0.5, 5],
                                         ridge.type.rates[ridge.type.rates$Threshold == 0.5, 5],
                                         elnet.type.rates[elnet.type.rates$Threshold == 0.5, 5]),
                        
                        Type.2.Test  = c(lasso.type.rates[lasso.type.rates$Threshold == 0.5, 6],
                                         ridge.type.rates[ridge.type.rates$Threshold == 0.5, 6],
                                         elnet.type.rates[elnet.type.rates$Threshold == 0.5, 6]))
theta5.df

# Question 5
lasso.thresh.train <- lasso.type.rates[which.min(lasso.type.rates$Difference.train),1]
ridge.thresh.train <- ridge.type.rates[which.min(ridge.type.rates$Difference.train),1]
elnet.thresh.train <- elnet.type.rates[which.min(elnet.type.rates$Difference.train),1]

lasso.thresh.test <- lasso.type.rates[which.min(lasso.type.rates$Difference.test),1]
ridge.thresh.test <- lasso.type.rates[which.min(ridge.type.rates$Difference.test),1]
elnet.thresh.test <- lasso.type.rates[which.min(elnet.type.rates$Difference.test),1]

lasso.thresh.train
ridge.thresh.train
elnet.thresh.train

lasso.thresh.test
ridge.thresh.test
elnet.thresh.test

lasso.diff.train <- lasso.type.rates[which.min(lasso.type.rates$Difference.train),4]
ridge.diff.train <- ridge.type.rates[which.min(lasso.type.rates$Difference.train),4]
elnet.diff.train <- elnet.type.rates[which.min(lasso.type.rates$Difference.train),4]

lasso.diff.test <- lasso.type.rates[which.min(lasso.type.rates$Difference.test),7]
ridge.diff.test <- ridge.type.rates[which.min(lasso.type.rates$Difference.test),7]
elnet.diff.test <- elnet.type.rates[which.min(lasso.type.rates$Difference.test),7]

approxEqual.df <- data.frame(Method = method,
                             Threshold.Train= c(lasso.thresh.train,
                                                ridge.thresh.train,
                                                elnet.thresh.train),
                             
                             Min.Diff.Train = c(lasso.diff.train,
                                                ridge.diff.train,
                                                elnet.diff.train),
                             
                             Threshold.Test = c(lasso.thresh.test,
                                                ridge.thresh.test,
                                                elnet.thresh.test),
                             
                             Min.Diff.Test  = c(lasso.diff.test,
                                                ridge.diff.test,
                                                elnet.diff.test))


approxEqual.df
