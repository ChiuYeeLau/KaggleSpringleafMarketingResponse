# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(randomForest)

set.seed(616)

cat("reading the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")
store <- read_csv("../input/store.csv")

train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]

# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,6,8:12,14:19)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
        if (class(train[[f]])=="character") {
                levels <- unique(c(train[[f]], test[[f]]))
                train[[f]] <- as.integer(factor(train[[f]], levels=levels))
                test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
        }
}

# imputation to the mean (assuming 0 is missing value not lack of competition)
# train$CompetitionOpenSinceMonth[train$CompetitionOpenSinceMonth==0] <- mean(train$CompetitionOpenSinceMonth, na.rm=TRUE)
# train$CompetitionOpenSinceYear[train$CompetitionOpenSinceYear==0] <- mean(train$CompetitionOpenSinceYear, na.rm=TRUE)
#train$Promo2SinceWeek[train$Promo2SinceWeek==0] <- mean(train$Promo2SinceWeek, na.rm=TRUE)
# train$Promo2SinceYear[train$Promo2SinceYear==0] <- mean(train$Promo2SinceYear, na.rm=TRUE)

#test$CompetitionOpenSinceMonth[test$CompetitionOpenSinceMonth==0] <- mean(test$CompetitionOpenSinceMonth, na.rm=TRUE)
#test$CompetitionOpenSinceYear[test$CompetitionOpenSinceYear==0] <- mean(test$CompetitionOpenSinceYear, na.rm=TRUE)
#test$Promo2SinceWeek[test$Promo2SinceWeek==0] <- mean(test$Promo2SinceWeek, na.rm=TRUE)
#test$Promo2SinceYear[test$Promo2SinceYear==0] <- mean(test$Promo2SinceYear, na.rm=TRUE)

cat("checking all stores are accounted for\n")
length(unique(train$Store))

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)

clf <- randomForest(train[,feature.names], 
                    log(train$Sales+1),
                    mtry=5,
                    ntree=17,
                    sampsize=225000,
                    # strata=train$Store,
                    do.trace=TRUE)

cat("model stats\n")
clf
cat("print model\n")
print(clf)
cat("Importance 1\n")
importance(clf)
cat("Permutation Importance Unscaled\n")
importance(clf, type = 1, scale = FALSE)
cat("GINI Importance\n")
importance(clf, type = 2)
cat("Plot Model\n")
plot(clf)
cat("Plot Importance\n")
plot(importance(clf), lty=2, pch=16)


cat("Predicting Sales\n")

pred <- exp(predict(clf, test)) -1
submission <- data.frame(Id=test$Id, Sales=pred)

cat("saving the submission file\n")
write_csv(submission, "rf1.csv")