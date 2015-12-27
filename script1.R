library(ggplot2)
library(caret)
library(RRF)

rm(list=ls())
setwd("~/R/DSS-PML/")

training <- read.csv("./input/pml-training.csv", stringsAsFactors = F, na.strings = c("NA", "", " "))
testing <- read.csv("./input/pml-testing.csv", stringsAsFactors = F, na.strings = c("NA", "", " "))

col2rem <- sapply(testing, function(x) {sum(is.na(x)) > 0})
col2rem[1:7] <- TRUE # removing "X", user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"
training <- training[, !col2rem]
testing <- testing[, !col2rem]

training$classe <- as.factor(training$classe)

lab <- ncol(training)

#install.packages("corrplot")
library(corrplot)
tcor <- cor(training[,-lab])
corrplot(tcor, type="lower", order="hclust", tl.col="black",tl.srt=45,  tl.cex = 0.7)#


set.seed(777)
inTrain <- createDataPartition(training$classe, p = 0.75, list = FALSE)
train <- training[inTrain, ]
valid <- training[-inTrain, ]
dim(train); dim(valid)
flagReg <- 1
set.seed(777)
tunerrf <- tuneRRF(train[, -lab], train[, lab], stepFactor=1.4, ntreeTry=250, improve=0.01,
                   trace=T, plot=F, doBest=F, flagReg=flagReg)
tmtry = tunerrf[which.min(tunerrf[,2]),1]

qplot(mtry, OOBError, data=as.data.frame(tunerrf), geom = c("point", "path"))+ theme_grey(base_size = 14) 

rrfmod1 <- RRF(train[, -lab], train[, lab], ntree=500, mtry = tmtry, 
               importance = T, localImp=F, flagReg=flagReg, coefReg = 0.8)
#rrfmod2 <- RRF(valid[, -lab], valid[, lab], ntree=250, mtry = tunerrf[which.min(tunerrf[,2]),1], 
#               importance = T, localImp=F, flagReg=flagReg, coefReg = 0.8)

predrrf1 <- predict(rrfmod1, valid[,-lab], type="class")
#predrrf2 <- predict(rrfmod2, train[,-lab], type="class")
aa1 <- confusionMatrix(predrrf1, valid[, lab])$overall['Accuracy']
#aa2 <- confusionMatrix(predrrf2, train[, lab])$overall['Accuracy']
cat("Average accuracy", aa1) #, mean(aa1, aa2))

RRF::varImpPlot(rrfmod1)
#varImpPlot(rrfmod2)
impRF1 <- rrfmod1$importance
#impRF2 <- rrfmod2$importance
vars1 <- rownames(impRF1)
#vars2 <- rownames(impRF2)
vimp1 <- data.frame(vars1, impRF1, row.names = NULL)
#vimp2 <- data.frame(vars2, impRF1, row.names = NULL)
MDG1<- as.character(vimp1[vimp1$MeanDecreaseGini>0 & order(vimp1$MeanDecreaseGini, decreasing = T), "vars1"])
#MDG2<- as.character(vimp2[vimp2$MeanDecreaseGini>0 & order(vimp2$MeanDecreaseGini, decreasing = T), "vars2"])

#cat("Selected variables:", paste(MDG1, collapse = ", "))

training <- training[, c(MDG1, "classe")]
testing <- testing[, c(MDG1, "problem_id")]
lab <- ncol(training)

#rm(list=c("train", "valid", "tunerrf", "vimp1", "tcor", "inTrain", 
#          "impRF1", "col2rem", "predrrf1", "rrfmod1", "vars1", "aa1"))

fitMod <- function(xtrain, ytrain, xtest, ytest, method){
  #returns prediction vector
  if (method=="rf"){
    require(randomForest, quietly=T)
    fit <- randomForest(x=xtrain, y=ytrain, mtry=tmtry)
    pred  <- predict(fit, newdata=xtest)
    
  }else if(method=="svm"){
    require(e1071, quietly=T)
    fit <- svm(x=xtrain, y=ytrain, probability = FALSE)
    pred <- predict(fit, xtest)
    
  }else if(method=="earth"){
    require(earth, quietly=T)
    fit <- earth(x=xtrain, y=ytrain) # build model
    pred <- as.numeric(predict(fit, newdata = xtest, type = "class"))
    
  }else if(method=="C50"){
    require(C50, quietly=T)
    fit <- C5.0(x=xtrain, y=ytrain) # build model
    pred <- predict(fit, newdata = xtest, type = "class")
    
  }else if(method=="RRF"){
    require(RRF, quietly=T)
    fit <- RRF(x=xtrain, y=ytrain, mtry=tmtry, flagReg=flagReg, coefReg = 0.8)
    pred  <- predict(fit, newdata=xtest, type = "class")
    
  }else if(method=="gbm"){
    require(gbm, quietly=T)
    fit <- gbm.fit(x=xtrain, y=ytrain, distribution ="multinomial", interaction.depth=6, shrinkage=0.1, n.trees=40, verbose = FALSE)
    p <- predict(fit, newdata=xtest, n.trees=40, type="response")
    pred <- factor(apply(p, 1, which.max), levels=c(1:5), labels = c("A", "B", "C", "D", "E"))
    
  }else if(method=="knn"){
    require(class, quietly=T)
    pred <- knn(train =xtrain, test = xtest, cl= ytrain) # build model
    
  }else if(method=="ipred"){
    require(ipred, quietly=T)
    train <- cbind(xtrain, classe=ytrain)
    fit <- bagging(classe~., data=train) # build model
    pred <- predict(fit, newdata = xtest, type="class")      
    
  }else{
    cat("Unknown method! Aborting")
    pred <- NA
  }
  
  #pred <- factor(pred, levels=c(1:5), labels=c("A","B","C","D","E"))
  names(pred) <- method
  #  cat("Accuracy of ", method, " for this fold is:", confusionMatrix(as.numeric(pred), as.numeric(ytest))$overall['Accuracy'], "\n")
  
  return(pred)
}
k <-5
set.seed(777)
folds <- createFolds(training[, lab], k = k, list = FALSE)
pred.df <- NULL
models <- c("rf", "svm", "C50", "gbm", "RRF", "knn", "ipred")
library(foreach)
library(doParallel)
cl <- makeCluster(8)
registerDoParallel(cl)
pred.df <- foreach(i = 1:k, 
                   .packages=c("caret"), 
                   #.export=c("folds","fitMod", "tmtry"), #export=ls(envir=globalenv())
                   .combine = rbind) %dopar% {
                     
                     #models <- c("rf", "svm", "earth", "C50", "party", "mda", "knn", "ipred")
                     sapply(models, function(x){
                       cat("Fitting ", x, " model \n")
                       fitMod(xtrain=training[folds!=i,-lab], ytrain=training[folds!=i,lab], xtest=training[folds==i,-lab],  ytest= training[folds==i,lab], method = x)})
                     #prediction <- fitMod(xtrain=cv.train[,-lab], ytrain=cv.train[,lab], xtest=cv.test[,-lab], ytest= cv.test[, lab], method = "glm")
                   }

train.cv <- NULL
for (n in 1:k){
  train.cv <- rbind(train.cv, training[folds==n,])
}
rownames(train.cv) <- NULL

pred.df <- as.data.frame(pred.df)
#pred.df$blended <- (apply(pred.df, 1, function(idx) which.max(tabulate(idx))))
pred.df$blended <- factor(apply(data.frame(lapply(pred.df, as.numeric), stringsAsFactors=FALSE), 1, function(idx) which.max(tabulate(idx))), 
                          levels=c(1:5), labels=c("A", "B", "C", "D", "E"))
#pred.df$classe <- train.cv$classe
rownames(pred.df) <- NULL

for (i in 1:(ncol(pred.df)-1)){
  cat("CV accuracy of", names(pred.df)[i], "is", confusionMatrix(pred.df[,i], train.cv$classe)$overall['Accuracy'], "\n")
}

# making the CV stacking model
straining <- data.frame(pred.df,train.cv)
slab <- ncol(straining)
set.seed(777)
k=5

spred.v <- NULL
ytest <- NULL
for (i in 1:k){
  stackedpred <- fitMod(xtrain=straining[folds!=i,-slab], ytrain=straining[folds!=i,slab], xtest=straining[folds==i,-slab],  ytest= straining[folds==i,slab], method = "gbm")
  spred.v <- c(spred.v, stackedpred)
  ytest <-c(ytest, straining[folds==i,slab])
}

spred.v <- factor(spred.v, levels=c(1:5), labels=c("A", "B", "C", "D", "E"))
ytest <- factor(ytest, levels=c(1:5), labels=c("A", "B", "C", "D", "E"))
#stacked prediction accuracy
cat("CV accuracy of stacked model is", confusionMatrix(spred.v, ytest)$overall['Accuracy'], "\n")

#stacked model is what we go for

# making prediction on test set
test.df <- sapply(models, function(x){
  cat("Fitting ", x, " model \n")
  fitMod(xtrain=training[,-lab], ytrain=training[,lab], xtest=testing[,-lab],  ytest= NULL, method = x)})

rownames(test.df) <- NULL

test.df <- as.data.frame(test.df)
test.df$blended <- factor(apply(data.frame(lapply(test.df, as.numeric), stringsAsFactors=FALSE), 1, function(idx) which.max(tabulate(idx))), 
                          levels=c(1:5), labels=c("A", "B", "C", "D", "E"))
stesting <- data.frame(test.df,testing)
tlab <- ncol(stesting)
set.seed(777)

#making prediction on the stacked model
finalpred <- fitMod(xtrain=straining[ ,-slab], ytrain=straining[,slab], xtest=stesting[,-tlab],  ytest= NULL, method = "gbm")

answers <- as.character(finalpred)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)

registerDoSEQ() -> unregister
stopCluster(cl)