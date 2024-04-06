---
title: "MACHINE LEARNING CASE-MOVIE RATING PREDICTION"
output:
word_document: default
pdf_document: default
date: "2023-12-02"
---
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(dplyr)
library(anytime)
library(lubridate)
library(ggplot2)

###DOWNLOADING "movielens/ml-10m.zip" and unzipping to "ratings" and "movies" files ###

options(timeout = 120)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.character(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))


movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::",3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.character(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

##################### Check download "ratings" and "movies"
glimpse(ratings)
glimpse(movies)
#################################### Join "ratings" and "movies" to "movielens" #####
movielens <- left_join(ratings, movies, by = "movieId")
glimpse(movielens)
####looking for incomplete entries (missing data)
sum(is.na(movies))
sum(is.na(ratings))
sum(is.na(movielens))

### "final_holdout_test" set, for validation, will be 10% of Movielens data, "edx" will be used for the training and test

set.seed(1, sample.kind = "Rounding") #  actual version R 4.2.3

test_index <- caret::createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final_holdout_test set are in "edx" set

final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final_holdout_test back to edx set

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

###looking for incomplete entries (missing data)
sum(is.na(final_holdout_test))  
sum(is.na(edx))

##################### TRAIN AND VALIDATION SETS OVERVIEW key figures, TYPE MODIF ######################
glimpse(edx)
glimpse(final_holdout_test)
summary(edx)
summary(final_holdout_test)

#how many different users
length(unique(edx$userId))
#how many different movies
length(unique(edx$movieId))

##formats changing
edx$userId <- as.factor(edx$userId) # change `userId` to `factor`.
edx$movieId <- as.factor(edx$movieId) # change `movieId` to `factor`.
edx$genres <- as.factor(edx$genres) # change `genres` to `factor`.
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01") # change `timestamp to `POSIXct`
edx <- edx %>% mutate(year_rate = year(timestamp)) #  add year OF rating from timestamp.
glimpse(edx)

#the most rated title 
Firsttitle <- edx%>% select(title,userId)%>%
  group_by(title)%>%summarise(nbuser=n())%>%
  arrange(desc(nbuser))
head(Firsttitle)

# Rating top down
distrib <- edx %>% select(rating,userId)%>%
  group_by(rating)%>%summarise(nbuser=n())%>%
  select(rating,nbuser)%>%
  arrange(desc(nbuser))
head(distrib,10)

# Rating distribution using geom_bar
rat <- as.data.frame <- edx %>% select(rating,userId)%>%
  group_by(rating)%>%summarise(nbusers=n())%>%
  select(rating,nbusers) %>% mutate(rating=as.character(rating))
ggplot(rat,aes(x=factor(rating),y=nbusers))+
  geom_bar(stat='Identity',fill=rat$rating)

# year_rate plot
table(unique(edx$year_rate))
edx %>% ggplot(aes(year_rate)) +
  geom_histogram(binwidth=1,color="darkblue",fill="#56B4E9") +
  labs(title = "Distribution of the year_rate",
       subtitle = "Ratings per year",
       x = "Year",
       y = "Frequency")

# Rating distribution top down
distyear <- edx %>% select(userId,year_rate)%>%
  group_by(year_rate)%>%summarise(yearcount=n())%>%
  arrange(desc(yearcount))
head(distyear,15)
sum(distyear$yearcount)

# Genres distribution 
distgen <- edx %>%
  select(genres, rating) %>%
  group_by(genres) %>%
  summarize(avg = mean(rating), nbpg = n())
distgen%>% arrange(desc(avg)) %>% head(20)
distgen%>% arrange(desc(nbpg)) %>% head(20)

# Ratings versus single genre
drama<-sum(grepl("Drama",edx$genres))
comedy<-sum(grepl("Comedy",edx$genres))
thriller<-sum(grepl("Thriller",edx$genres))
romance<-sum(grepl("Romance",edx$genres))

#### TABLE GENRE VS Nb RATINGS
GenTable1 <-data.frame(
  RATINGS=c(drama,comedy,thriller,romance),
  GENRE=c("Drama","Comedy","Thriller","Romance"))
glimpse(GenTable1)

#### chart GENRE VS Nb RATINGS
ggplot(GenTable1,aes(x ="",y=RATINGS,fill=GENRE,labels=GENRE))+
  geom_bar(width = 1, stat = "identity")+
  coord_polar("y", start=0)+
  scale_fill_brewer(palette="Blues")+
  theme_minimal()+
  theme(
    axis.title.y = element_blank(),
    panel.border = element_blank(),
    panel.grid=element_blank(),
    axis.ticks = element_blank(),
    plot.title=element_text(size=14, face="bold"))

#Partition training test
edx <- edx %>% select(userId, movieId, rating)
test_index <- createDataPartition(edx$rating, times = 1, p = .2, list = F)# Create the index
training <- edx[-test_index, ] # Create Train set
test <- edx[test_index, ] # Create Test set
test <- test %>% # The same movieId and usersId appears in both set. (Not the same cases)
  semi_join(training, by = "movieId") %>%
  semi_join(training, by = "userId")

rmse_table <-0 
head(rmse_table)
glimpse(edx)
glimpse(test)
dim(training)
dim(test)
dim(edx)


############### baselineMODEL
mu <- mean(training$rating) # Mean accross all movies.
model0_RMSE <- RMSE(test$rating, mu) # RMSE in test set.
model0_RMSE

rmse_table <- data_frame(Method = "Baseline MODEL", RMSE = model0_RMSE)
rmse_table %>% knitr::kable(caption = "RMSEs")

############# Movie effect Model
Mu <- mean(training$rating)

movieAVG <- training %>%mutate(movieId=as.factor(movieId))%>%
  group_by(movieId) %>%
  summarize(Mi = mean(rating - Mu))

PREDICTIONS <- test %>%
  left_join(movieAVG, by = "movieId") %>%
  mutate(pred = Mu + Mi) %>% .$pred

model1_RMSE <- RMSE(PREDICTIONS,test$rating)
model1_RMSE  
rmse_table <- rbind(rmse_table, data_frame(Method = "Movie Effect", RMSE = model1_RMSE))
rmse_table %>% knitr::kable(caption = "RMSEs")

############ User effect Model
Mu <- mean(training$rating)

userEFFAVG <- training %>%mutate(userId=as.factor(userId))%>%
  group_by(userId) %>%
  summarize(Mi = mean(rating - Mu))

PREDICTIONS <- test %>%
  left_join(userEFFAVG, by = "userId") %>%
  mutate(pred = Mu + Mi) %>% .$pred

model2_RMSE <- RMSE(PREDICTIONS,test$rating)
model2_RMSE
rmse_table <- rbind(rmse_table, data_frame(Method = "User Effect", RMSE = model2_RMSE))
rmse_table %>% knitr::kable(caption = "RMSEs")

######## User and Movie effect Model
Mu <- mean(training$rating)

movieAVG <- training %>%mutate(movieId=as.factor(movieId))%>%
  group_by(movieId) %>%
  summarize(Mi = mean(rating - Mu))

userAVG <- training %>% mutate(movieId=as.factor(movieId))%>%
  left_join(movieAVG, by = "movieId") %>%
  group_by(userId) %>%
  summarize(Ui = mean(rating - Mu - Mi))

PREDICTIONS <- test %>%
  left_join(movieAVG, by = "movieId") %>%
  left_join(userAVG, by = "userId") %>%
  mutate(pred = Mu + Mi + Ui) %>% .$pred

modelMU_RMSE <- RMSE(PREDICTIONS,test$rating)
modelMU_RMSE  
rmse_table <- rbind(rmse_table,data_frame(Method = "User & Movie Effect", RMSE = modelMU_RMSE))
rmse_table %>% knitr::kable(caption = "RMSEs")

########## Regularization
lambda_values <- seq(3,6,.5)

RMSE_function_reg <- sapply(lambda_values, function(l){
  
  mu <- mean(training$rating)
  
  Mi <- training %>%
    group_by(movieId) %>%
    summarize(Mi = sum(rating - mu)/(n()+l))
  
  Ui <- training %>%
    left_join(Mi, by="movieId") %>%
    group_by(userId) %>%
    summarize(Ui = sum(rating - Mi - mu)/(n()+l))
  
  predicted_ratings <- test %>%
    left_join(Mi, by = "movieId") %>% 
    left_join(Ui, by = "userId") %>%
    mutate(pred = mu + Mi + Ui) %>% .$pred
  
  return(RMSE(predicted_ratings, test$rating))
})

qplot(lambda_values, RMSE_function_reg,
      main = "Regularisation",
      xlab = "Lambda", ylab ="RMSE" ) # lambda vs RMSE
lambda_opt <- lambda_values[which.min(RMSE_function_reg)]# Lambda which minimizes RMSE
lambda_opt 
rmse_table <- rbind(rmse_table, 
                    data_frame(Method = "User & Movie Effect Regularisation, test set ",
                               RMSE = min(RMSE_function_reg)))
rmse_table %>% knitr::kable(caption = "RMSEs")

###final_holdout_test TYPES
final_holdout_test <- final_holdout_test %>% 
  mutate( userId = as.factor(userId),
          movieId=as.factor(movieId))
glimpse(final_holdout_test)

###############  PREDICTIONS MODEL : User & Movie Effect on final_houldout_test
mu <- mean(training$rating)

movieAVG <- training %>%
  group_by(movieId) %>%
  summarize(Mi = mean(rating - mu))

userAVG <- training %>%
  left_join(movieAVG, by = "movieId") %>%
  group_by(userId) %>%
  summarize(Ui = mean(rating - mu- Mi))

PREDICTIONS <- final_holdout_test %>%
  left_join(movieAVG, by = "movieId") %>%
  left_join(userAVG, by = "userId") %>%
  mutate(pred = mu + Ui + Mi) %>% .$pred

val_RMSE <- RMSE(PREDICTIONS, final_holdout_test$rating, na.rm = TRUE)
val_RMSE

rmse_table_val <- data_frame(Method = "User & Movie Effect on final houldout test",
                             RMSE = val_RMSE)
rmse_table <- rbind(rmse_table,
                    data_frame(Method = "User & Movie Effect on final_houldout_test", RMSE = val_RMSE))

rmse_table %>% knitr::kable(caption = "RMSEs")

################ PREDICTIONS on final_holdout_test with parameters from edx set, model==regularized user & movie effect
lambda_values <- seq(3,6,.5)

RMSE_function_reg <- sapply(lambda_values, function(l){
  
  mu <- mean(edx$rating)
  
  Mi <- edx %>%
    group_by(movieId) %>%
    summarize(Mi = sum(rating - mu)/(n()+l))
  
  Ui <- edx %>%
    left_join(Mi, by="movieId") %>%
    group_by(userId) %>%
    summarize(Ui = sum(rating - Mi - mu)/(n()+l))
  
  predicted_ratings <- final_holdout_test %>%
    left_join(Mi, by = "movieId") %>% 
    left_join(Ui, by = "userId") %>%
    mutate(pred = mu + Mi + Ui) %>% .$pred
  
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

qplot(lambda_values, RMSE_function_reg,
      main = "Regularisation",
      xlab = "Lambda", ylab ="RMSE" ) # lambda vs RMSE

lambda_opt <- lambda_values[which.min(RMSE_function_reg)]
lambda_opt # Lambda which minimizes RMSE

rmse_table <- rbind(rmse_table,
                    data_frame(Method = "User & Movie Effect Regularisation with edx set",
                               RMSE = min(RMSE_function_reg)))
rmse_table %>% knitr::kable(caption = "RMSEs")
#########  END #############