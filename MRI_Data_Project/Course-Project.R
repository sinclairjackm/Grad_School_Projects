#All Libraries that we will need for data cleanup
library(eeptools)
library(plyr)
library(RCurl)
library(psych)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(ggcorrplot)
library(corrplot)
library(MASS)
library(e1071)
library(nnet)
library(class)
library(randomForest)
library(gbm)
library(broom)
library(pROC)


cross_init <- read.table(file = "oasis_cross-sectional.csv", sep = ",", header=T)
long_init <- read.table(file = "oasis_longitudinal.csv", sep = ",", header=T)
head(cross_init)
head(long_init)
cross <- subset(cross_init, select = -c(ID, Delay, Hand))
head(cross)
long <- subset(long_init, select = -c(Subject.ID, MRI.ID, Group, Visit, MR.Delay, Hand))
head(long)
summary(cross)
summary(long)
cross_final <- na.omit(cross)
long_final <- na.omit(long)

cross_final$M.F <- as.factor(cross_final$M.F)
cross_final$Section <- as.factor(cross_final$Section)
long_final$M.F <- as.factor(long_final$M.F)
long_final$Section <- as.factor(long_final$Section)
colnames(long_final)[3] = "Educ"
summry(cross_final)
summary(long_final)
full <- rbind(cross_final, long_final)
full <- transform(full, M.F = as.numeric(M.F), Section = as.numeric(Section))
summary(full)
sapply(full$M.F, mode)
###scatterplots
scatterplot1 <- ggplot(full, aes(y = CDR, x = M.F)) + geom_point()
scatterplot2 <- ggplot(full, aes(y = CDR, x = Age)) + geom_point()
scatterplot3 <- ggplot(full, aes(y = CDR, x = Educ)) + geom_point()
scatterplot4 <- ggplot(full, aes(y = CDR, x = SES)) + geom_point()
scatterplot5 <- ggplot(full, aes(y = CDR, x = MMSE)) + geom_point()
scatterplot6 <- ggplot(full, aes(y = CDR, x = eTIV)) + geom_point()
scatterplot7 <- ggplot(full, aes(y = CDR, x = nWBV)) + geom_point()
scatterplot8 <- ggplot(full, aes(y = CDR, x = ASF)) + geom_point()
scatterplot9 <- ggplot(full, aes(y = CDR, x = Section)) + geom_point()
grid.arrange(scatterplot1, scatterplot2, scatterplot3, scatterplot4, scatterplot5, scatterplot6, scatterplot7, scatterplot8, scatterplot9, ncol=2)
sum(full$M.F=='1')
sum(full$M.F=='2')
###correlogram
corrplot(cor(full))
cor(cross_final)
corrplot(cor(long_final))
cor(long_final)

model.matrix(~0+., data=full) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag=FALSE, type="lower", lab=TRUE, lab_size=2)

pairs(full)


n = dim(full)[1]
n1 = round(n/5)
set.seed(12345)
flag = sort(sample(1:n, n1))
fulltrain = full[-flag,]
fulltest = full[flag,]
head(full)
## Logistic Regression
model_base <- multinom(CDR ~ ., data=full)
summary(model_base)
# AIC of 661.985
model_i <- multinom(CDR ~ M.F + Age + Educ + SES + MMSE + nWBV + ASF + Section, data=fulltrain)
summary(model_i)
# AIC of 514.8518
model_ii <- multinom(CDR ~ M.F + Age + Educ + SES + MMSE + nWBV + ASF, data=fulltrain)
summary(model_ii)
# AIC: 516.8267
model_iii <- multinom(CDR ~ M.F + Age + Educ + SES + MMSE + nWBV + Section, data=fulltrain)
summary(model_iii)
# AIC: 515.0185
model_iv <- multinom(CDR ~ M.F + Age + Educ + MMSE + nWBV + ASF + Section, data=fulltrain)
summary(model_iv)
# AIC: 507.2386 
mean(predict(model_iv,fulltest[,c(1:3,5,8:10)]) != fulltest[,6])
# test error is 0.3070175

## KNN
kk <- 1:50
cverror <- NULL;
for (i in 1:length(kk)){
  ypred.test <- knn(fulltrain[,c(1:3,5,8:10)],fulltest[,c(1:3,5,8:10)],fulltrain[,6], k = kk[i]);
  temptesterror <- mean(knn(fulltrain[,c(1:3,5,8:10)],fulltest[,c(1:3,5,8:10)],fulltrain[,6], k = kk[i]) != fulltest[,6]);
  cverror <- c(cverror, temptesterror); 
}
cverror
#best at k=11, test error is 0.2368421

## LDA
model_lda <- lda(fulltrain[,c(1:3,5,8:10)], fulltrain[,6])
mean(predict(model_lda,fulltest[,c(1:3,5,8:10)])$class != fulltest[,6])
# test error is 0.2807018

## iii. Naive Bayes
model_nb <- naiveBayes(fulltrain[,c(1:3,5,8:10)], fulltrain[,6])
mean(predict(model_nb,fulltest[,c(1:3,5,8:10)]) != fulltest[,6])
# test error is 0.3421053



### Average preformance
B= 100
TEALL = NULL
n = dim(full)[1]
n1 = round(n/10)
set.seed(12345)
for (b in 1:B){
flag = sort(sample(1:n, n1))
fulltrain2 = full[-flag,]
fulltest2 = full[flag,]
## LDA
model2_lda <- lda(fulltrain2[,c(1:3,5,8:10)], fulltrain2[,6])
error_lda <- mean(predict(model2_lda,fulltest2[,c(1:3,5,8:10)])$class != fulltest2[,6])
## iii. Naive Bayes
model2_nb <- naiveBayes(fulltrain2[,c(1:3,5,8:10)], fulltrain2[,6])
error_nb <- mean(predict(model2_nb,fulltest2[,c(1:3,5,8:10)]) != fulltest2[,6])
## iv. Logisic Regression
model2_lr <- multinom(CDR ~ M.F + Age + Educ + MMSE + nWBV + ASF + Section, data=fulltrain2)
error_lr <- mean(predict(model2_lr,fulltest2[,c(1:3,5,8:10)]) != fulltest2[,6])
## v. KNN
model2_knn <- knn(fulltrain2[,c(1:3,5,8:10)],fulltrain2[,c(1:3,5,8:10)],fulltrain2[,6], k=11)
error_knn <- mean(knn(fulltrain2[,c(1:3,5,8:10)],fulltest2[,c(1:3,5,8:10)],fulltrain2[,6], k=11) != fulltest2[,6])
TEALL = rbind(TEALL,cbind(error_lda, error_nb, error_lr, error_knn) );
}
apply(TEALL, 2, mean)
sqrt(apply(TEALL, 2, var))

#error_lda  error_nb  error_lr error_knn 
#0.2773684 0.2522807 0.2508772 0.2733333 
#error_lda   error_nb   error_lr  error_knn 
#0.06291993 0.05685024 0.05221645 0.05576808


library(caret)
predicted <- predict(model_iv,fulltest[,c(1:3,5,8:10)])
confusionMatrix(predicted,as.factor(fulltest[,6]))


model_v <- multinom(CDR ~ M.F + MMSE + nWBV, data=fulltrain)
summary(model_v)

summary(full)
nrow(full)
library(gbm)
# vii. Boosting
model_gbm <-gbm(CDR ~ Age + Educ + MMSE + nWBV + ASF, data=fulltrain,
distribution = "bernoulli", n.trees = 10000, interaction.depth = 3)
print(model_gbm)
pred_y = predict.gbm(model_gbm, fulltest[,6])
x_ax = 1:length(pred_y)
plot(x_ax, fulltest[,6], col="blue", pch=20, cex=.9)
lines(x_ax, pred_y, col="red", pch=20, cex=.9)
n.trees = seq(from = 100, to = 10000, by = 100)
mean(predict(model_gbm,fulltest[,c(2,3,5,8:9)], n.trees=n.trees) != fulltest$CDR)
# test error is 0.0002564103


library(car)
library(pastecs)
library(corrplot)
library(leaps)
library(MASS)
library(lars)
library(pls)
model_LASSO <- lars( as.matrix(fulltrain[,c(1:5,7:10)]), fulltrain[,6], type= "lasso", trace= TRUE)
Cp1 <- summary(model_LASSO)$Cp
index1 <- which.min(Cp1)
lasso.lambda <- model_LASSO$lambda[index1];
fit_c_v <- predict(model_LASSO, fulltest[,-6], s=lasso.lambda, type="fit", mode="lambda");
yhat_c_v <- fit_c_v$fit;
mse_c_v <- mean( (yhat_c_v - fulltest[,6])^2);
mse_c_v
model_LASSO$beta[index1,]
summary(model_LASSO)

## iv
model_ridge <- lm.ridge(CDR ~ ., data = fulltrain, lambda= seq(0,100,0.001))
#need to select ridge regression model with the optimal lambda value, using method i and ii from class
select(model_ridge)
lambda_opt2 <- which.min(model_ridge$GCV)
#then finding coefficients and intercepts
rig_2_coef <- model_ridge$coef[,lambda_opt2]/model_ridge$scales
rig_2_coef
rig_2_intercepts <- -sum(rig_2_coef  * colMeans(fulltest[,c(1:5,7:10)] )  )+ mean(fulltest[,6])
pred_test2 <- as.matrix(fulltest[,c(1:5,7:10)]) %*% as.vector(rig_2_coef) + rig_2_intercepts;
mse_ridge <- mean((pred_test2 - fulltest[,6])^2); 
mse_ridge

## part (vi) PCR, using the pls library from class
model_PCR <- pcr(CDR ~., data=fulltrain, validation="CV");
validationplot(model_PCR)
summary(model_PCR)
ncompopt <-which.min(model_PCR$validation$adj)
ncompopt
## CV shows that 9 PC is the best
pred_PCR <- predict(model_PCR, ncomp = ncompopt, newdata = fulltest[,c(1:5,7:10)])
mse_PCR <- mean( (pred_PCR - fulltest[,6])^2);
mse_PCR
coef2 <- predict(model_PCR, ncomp = ncompopt, newdata = as.matrix(rbind(rep.int(0,17),rep.int(1,17))));
c(coef2[1],coef(model_PCR, ncomp = ncompopt))

## part (vii) PLS
model_PLS <- plsr(CDR ~., data=fulltrain, validation="CV")
summary(model_PLS)
ncompoptplsr <-which.min(model_PLS$validation$adj)
ncompoptplsr
## CV shows that 8 PC is the best
pred_PLS <- predict(model_PLS, fulltest[,c(1:5,7:10)], ncomp=ncompoptplsr)
mse_PLS <- mean((pred_PLS - fulltest[,6])^2)
mse_PLS
model_PLS
B= 100
TEALL = NULL
n = dim(full)[1]
n1 = round(n/10)
set.seed(12345)
for (b in 1:B){
flag = sort(sample(1:n, n1))
fulltrain2 = full[-flag,]
fulltest2 = full[flag,]

## iv
model_e_iv <- lm.ridge(CDR ~ ., data = fulltrain2, lambda= seq(0,100,0.001))
#need to select ridge regression model with the optimal lambda value, using method i and ii from class
select(model_e_iv)
lambda_opt2 <- which.min(model_e_iv$GCV)
#then finding coefficients and intercepts
rig_2_coef <- model_e_iv$coef[,lambda_opt2]/model_e_iv$scales
rig_2_intercepts <- -sum(rig_2_coef  * colMeans(fulltest2[,c(1:5,7:10)] )  )+ mean(fulltest2[,6])
pred_test2 <- as.matrix(fulltest2[,c(1:5,7:10)]) %*% as.vector(rig_2_coef) + rig_2_intercepts;
mse_e_iv <- mean((pred_test2 - fulltest2[,6])^2); 

## v
model_e_v <- lars(as.matrix(fulltrain2[,c(1:5,7:10)]), fulltrain2[,6], type= "lasso", trace= TRUE)
Cp2 <- summary(model_e_v)$Cp
index2 <- which.min(Cp2)
lasso.lambda2 <- model_e_v$lambda[index2];
fit_e_v <- predict(model_e_v, fulltest2[,-6], s=lasso.lambda2, type="fit", mode="lambda");
yhat_e_v <- fit_e_v$fit;
mse_e_v <- mean( (yhat_e_v - fulltest2[,6])^2);

## vi
model_e_vi <- pcr(CDR ~., data=fulltrain2, validation="CV");
ncompopt2 <-which.min(model_e_vi$validation$adj)
pred_e_vi <- predict(model_e_vi, ncomp = ncompopt2, newdata = fulltest2[,c(1:5,7:10)])
mse_e_vi <- mean((pred_e_vi - fulltest2[,6])^2);


## vii
model_e_vii <- plsr(CDR ~., data=fulltrain2, validation="CV")
ncompoptplsr2 <-which.min(model_e_vii$validation$adj)
pred_e_vii <- predict(model_PLS, fulltest2[,c(1:5,7:10)], ncomp=ncompoptplsr)
mse_e_vii <- mean((pred_e_vii - fulltest2[,6])^2)

## gathering all
TEALL = rbind(TEALL, cbind(mse_e_iv, mse_e_v, mse_e_vi, mse_e_vii))
}

dim(TEALL);
round(apply(TEALL, 2, mean),4)
round(apply(TEALL, 2, var),4)


## compare model_iii with others
t.test(TEALL[,2], TEALL[,1],paired=TRUE)
t.test(TEALL[,3], TEALL[,1],paired=TRUE)
t.test(TEALL[,4], TEALL[,1],paired=TRUE)
t.test(TEALL[,1], TEALL[,4],paired=TRUE)
t.test(TEALL[,2], TEALL[,4],paired=TRUE)
t.test(TEALL[,3], TEALL[,4],paired=TRUE)


