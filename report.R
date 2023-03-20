origin_data=read.csv("bank_personal_loan.csv")
install.packages("VIM")
library("VIM")
aggr(origin_data,prop=FALSE,numbers=TRUE)  # check missing values
summary(origin_data)

library(pheatmap)
corr <- cor(origin_data)
pheatmap(corr, display_numbers = TRUE)
data<-origin_data[, colnames(origin_data) != "Experience"]

library(randomForest)
library(caret)
library(pROC)

set.seed(123)

trainIndex <- createDataPartition(data$Personal.Loan, p = 0.7, list = FALSE, times = 1)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

rf_model_train<-randomForest(as.factor(train_data$Personal.Loan) ~ .,mtry=3,ntree=500,
                             data=train_data,importance=TRUE, proximity=TRUE)

plot(rf_model_train)
rf.test<-predict(rf_model_train,newdata = test_data,type = "class")
rf.cf<-caret::confusionMatrix(as.factor(rf.test),as.factor(test_data$Personal.Loan))
rf.cf
importance(rf_model_train)
pre_ran <- predict(rf_model_train,newdata=test_data)
obs_p_ran = data.frame(prob=pre_ran,obs=test_data$Personal.Loan)
table(test_data$Personal.Loan,pre_ran,dnn=c("Real Value","Predictive Value"))
ran_roc <- roc(test_data$Personal.Loan,as.numeric(pre_ran))
plot(ran_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='ROC Curve,mtry=3,ntree=500')

#Tuning
# Create model with default parameters
trControl <- trainControl(method="repeatedcv", number=10, repeats=1, search="grid")
tuneGrid <- expand.grid(mtry=c(1:11))
seed <- 111
set.seed(seed)

rf_grid <- train(x=train_data, y=as.factor(train_data$Personal.Loan), method="rf", 
                 tuneGrid = tuneGrid, 
                 metric="Kappa", #metric='Kappa'
                 ntree=500,
                 trControl=trControl)
rf_grid

rf_model_train_2<-randomForest(as.factor(train_data$Personal.Loan) ~ .,mtry=2,ntree=500,
                               data=train_data,importance=TRUE, proximity=TRUE)

plot(rf_model_train_2)
rf.test<-predict(rf_model_train_2,newdata = test_data,type = "class")
rf.cf<-caret::confusionMatrix(as.factor(rf.test),as.factor(test_data$Personal.Loan))
rf.cf

pre_ran <- predict(rf_model_train_2,newdata=test_data)
obs_p_ran = data.frame(prob=pre_ran,obs=test_data$Personal.Loan)
table(test_data$Personal.Loan,pre_ran,dnn=c("Real Value","Predictive Value"))
ran_roc <- roc(test_data$Personal.Loan,as.numeric(pre_ran))
plot(ran_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='ROC Curve,mtry=2,ntree=500')

