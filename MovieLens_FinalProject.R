##### HEADING #####

#### HarvardX Data Science Professional Certificate Program Capstone
# Title:  MovieLens Project - Creating a Film Recommendation Algorithm
# Author: Andrew Infantino
# Date:   11 August 2020

##### INSTALLATION #####
#### Install or load all packages necessary for project:
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages('lubridate', 
                                         repos = "http://cran.us.r-project.org")

#### Download MovieLens dataset as temp file:
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), 
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

#### Create dataset for analysis in R 3.6.2:
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId], 
         title = as.character(title), 
         genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

#### Create validation set with 10% of MovieLens data for 
#### testing model:
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

#### Include rows removed from validation set in edx set for
#### training final model (post-development):
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#### Remove all data other than validation and edx sets:
rm(dl, ratings, movies, test_index, temp, movielens, removed)


##### RMSE & DEVELOPMENT SETS #####

#### Define RMSE function for testing and developing model:
# RMSE >> 1 --> error by at least a star.
# Target: RMSE < 0.86490
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### Create training and testing sets for developing model:
set.seed(1, sample.kind="Rounding")     # Obtain consistent results.
index <- createDataPartition(y = edx$rating, times = 1, 
                             p = 0.2, list = FALSE)
train <- edx[-index,]                   # 80% of edx data.
test <- edx[index, ] %>%                # 20% of edx data.
  semi_join(train, by = "movieId") %>%  # Exclude users and 
  semi_join(train, by = "userId")       # movies not in train.

#### Accomodate train, test, edx, and validation sets for 
#### time effects and round dates to the nearest week:
train <- mutate(train, date = as_datetime(timestamp)) %>%
  mutate(week = round_date(date, unit = 'week'))
test <- mutate(test, date = as_datetime(timestamp)) %>%
  mutate(week = round_date(date, unit = 'week'))
edx <- mutate(edx, date = as_datetime(timestamp)) %>%
  mutate(week = round_date(date, unit = 'week'))
validation <- mutate(validation, date = as_datetime(timestamp)) %>%
  mutate(week = round_date(date, unit = 'week'))


##### SIMPLE PREDICTION #####

#### Simple prediction parameters:
x <- train$rating      # Simple variable for train set ratings.
mu_hat <- mean(x)      # Average train set rating.
st_dev <- sqrt(sum((x-mu_hat)^2) / length(x)) # Standard deviation of train set ratings.
simple_norm <- mean(x >= (mu_hat-st_dev) &    # Proportion of train set ratings within
                      x <= (mu_hat+st_dev))   # a standard deviation from the mean.

#### Print simple prediction parameters for user:
cat("Average rating = ", mu_hat, " stars.")
cat("Standard deviation = ", st_dev, " stars.")
cat(100*simple_norm,"% of 'train' ratings range from ", 
    mu_hat-st_dev, "to ", mu_hat+st_dev, " stars.")

#### Print top 5 most rated train films scoring 2.5:
train %>%
  group_by(movieId) %>%
  filter(rating == 2.5) %>%
  summarize(N = n(), title[1]) %>%
  top_n(5, N) %>%
  arrange(desc(N))

#### Print top 5 most rated train films scoring 4.5:
train %>%
  group_by(movieId) %>%
  filter(rating == 4.5) %>%
  summarize(N = n(), title[1]) %>%
  top_n(5, N) %>%
  arrange(desc(N))

#### Plot histogram of train set rating distribution:
train %>%
  ggplot(aes(rating)) +
  xlab("Film rating levels") +
  ylab("Number of ratings") +
  ggtitle("Film rating distribution (train)") +
  geom_histogram(binwidth = 0.5, color = "black")

#### Create data frame to store RMSE results of model tests:
# The frame will be accumulatively mutated over time.
RMSE_results <- data.frame(Method = "Simple average",
                           RMSE = RMSE(test$rating, mu_hat))
RMSE_results  # Print all RMSE results.


##### IRIZARRY'S MODEL #####
#### Define film and user effect  variables and predictions
#### for Irizarry's model:
b_i_hat <- train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))
b_u_hat <- train %>%
  left_join(b_i_hat, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))
rai_predictions <- test %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

#### Calculate RMSE results of Irizarry's model:
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Film and user effects",
                                     RMSE = RMSE(test$rating, rai_predictions)))
RMSE_results  # Print all RMSE results.

##### TIME EFFECT #####
#### Plot train ratings against weeks of rating submission:
train %>%
  group_by(week) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(week, rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Week of rating submission") +
  ylab("Rating") +
  ggtitle("Film ratings over time")

#### Define variable and predictions for time effects in 
#### addition to Irizarry's model. Calculate and print results:
b_d_hat <- train %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  group_by(week) %>%
  summarize(b_d = mean(rating - mu_hat - b_i - b_u))
date_predictions <- test %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  left_join(b_d_hat, by = 'week') %>%
  mutate(pred = mu_hat + b_i + b_u + b_d) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Film, user, and time effects",
                                     RMSE = RMSE(test$rating, date_predictions)))
RMSE_results

##### GENRE EFFECT #####

#### Graph the average rating (with standard deviations) of 
#### each distinct genre and genre combination (genres) in
#### train set:
train %>% 
  group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), 
            se = sd(rating)/sqrt(n())) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, 
             ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.x=element_blank()) +
  ylab("Average rating") +
  ggtitle("Film ratings by distinct genres and genre combinations")

#### Print number of distinct train genres:
n_distinct(train$genres)

#### Define variable and predictions for genre effects in 
#### addition to Irizarry's model. Calculate and print results:
b_g_hat <- train %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))
genre_predictions <- test %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  left_join(b_g_hat, by = 'genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Film, user, and genre effects",
                                     RMSE = RMSE(test$rating, genre_predictions)))
RMSE_results

#### Re-calculate and re-predict genre effects in addition to
#### Irizarry's model and date effects. Calculate and print results:
b_g_hat <- train %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  left_join(b_d_hat, by = 'week') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u - b_d))
futg_predictions <- test %>%
  left_join(b_i_hat, by = 'movieId') %>%
  left_join(b_u_hat, by = 'userId') %>%
  left_join(b_d_hat, by = 'week') %>%
  left_join(b_g_hat, by = 'genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_d + b_g) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Film, user, time, and genre effects",
                                     RMSE = RMSE(test$rating, futg_predictions)))
RMSE_results

#### Replot average train rating of genres that have only at
#### least 1000 ratings:
train %>% 
  group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), 
            se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, 
             ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.x=element_blank()) +
  ylab("Average rating") +
  ggtitle("Film ratings by genres with at least 1000 ratings")

##### REGULARIZATION #####

#### Create data frame of non-duplicate film IDs and titles:
film_titles <- train %>%
  select(movieId, title) %>%
  distinct()

#### Rank top recommendations of Irizarry's model:
train %>%
  count(movieId) %>%
  left_join(b_i_hat, by = "movieId") %>%
  left_join(film_titles, by = "movieId") %>%
  mutate(rating = mu_hat + b_i) %>%
  arrange(desc(b_i)) %>%
  select(title, rating, count = n) %>%
  slice(1:10)

#### Determine appropriate tuning parameter for Irizarry's
#### regularized model:
lambdas <- seq(0, 10, 0.25)
rmse_cv <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n()+l))
  predictions <- test %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predictions, test$rating))
})

#### Plot distribution of lambdas tested for Irizarry's
#### regularized model. Print tuning parameter with lowest
#### RMSE for user:
qplot(lambdas, rmse_cv)
cat("Tuning parameter for regularized film and user effects = ", 
    lambdas[which.min(rmse_cv)])

#### Define lambda as parameter selected above. Code Irizarry's
#### regularized film and user effect variables. Code predictions
#### of Irizarry's regularized model. Calculate and print RMSE
#### results:
lambda <- lambdas[which.min(rmse_cv)]
b_i_reg <- train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
rai_reg_pred <- test %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Regularized film and user effects",
                                     RMSE = RMSE(test$rating, rai_reg_pred)))
RMSE_results

##### GENRE REGULARIZATION #####

#### Determine and print new tuning parameter for regularized
#### film, user, and genre effects:
lambdas <- seq(0, 10, 0.25)
rmse_cv <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n()+l))
  b_g <- train %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+l))
  predictions <- test %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_g, by = 'genres') %>%
    mutate(pred = mu_hat + b_i + b_u + b_g) %>%
    pull(pred)
  return(RMSE(predictions, test$rating))
})
cat("Tuning parameter for regularized film, user, and genre effects = ", 
    lambdas[which.min(rmse_cv)])

#### Recode lambda and regularized variables for film, user,
#### and genre effects. Note, the former two effects are re-
#### calculated with respect to new lambda in order to obtain
#### correct model predictions:
lambda <- lambdas[which.min(rmse_cv)]
b_i_reg <- train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
b_g_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+lambda))
genre_reg_pred <- test %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_g_reg, by = 'genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred

#### Calculate RMSE of regularized film/user/genre effects
#### model. Print all RMSE results:
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Regularized film, user, and genre effects",
                                     RMSE = RMSE(test$rating, genre_reg_pred)))
RMSE_results

##### TIME REGULARIZATION #####

#### Determine and print new tuning parameter for regularized
#### film, user, and date effects:
lambdas <- seq(0, 10, 0.25)
rmse_cv <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n()+l))
  b_d <- train %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(week) %>%
    summarize(b_d = sum(rating - mu_hat - b_i - b_u)/(n()+l))
  predictions <- test %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_d, by = 'week') %>%
    mutate(pred = mu_hat + b_i + b_u + b_d) %>%
    pull(pred)
  return(RMSE(predictions, test$rating))
})
cat("Tuning parameter for regularized film, user, and time effects = ", 
    lambdas[which.min(rmse_cv)])

#### Code new lambda and regularized variables for film, user,
#### and date effects. Calculate predictions and print RMSE:
lambda <- lambdas[which.min(rmse_cv)]
b_i_reg <- train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
b_d_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  group_by(week) %>%
  summarize(b_d = sum(rating - mu_hat - b_i - b_u)/(n()+lambda))
date_reg_pred <- test %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_d_reg, by = 'week') %>%
  mutate(pred = mu_hat + b_i + b_u + b_d) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Regularized film, user, and time effects",
                                     RMSE = RMSE(test$rating, date_reg_pred)))
RMSE_results

##### FINAL REGULARIZATION #####

#### Determine and print new tuning parameter for regularized
#### film, user, date, and genre effects:
lambdas <- seq(0, 10, 0.25)
rmse_cv <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n()+l))
  b_d <- train %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(week) %>%
    summarize(b_d = sum(rating - mu_hat - b_i - b_u)/(n()+l))
  b_g <- train %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_d, by = 'week') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u - b_d)/(n()+l))
  predictions <- test %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_d, by = 'week') %>%
    left_join(b_g, by = 'genres') %>%
    mutate(pred = mu_hat + b_i + b_u + b_d + b_g) %>%
    pull(pred)
  return(RMSE(predictions, test$rating))
})
cat("Tuning parameter for regularized film, user, time, and genre effects = ", 
    lambdas[which.min(rmse_cv)])

#### Code new lambda and regularized variables for film, user,
#### date, and genre effects. Calculate predictions and print RMSE:
lambda <- lambdas[which.min(rmse_cv)]
b_i_reg <- train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
b_d_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  group_by(week) %>%
  summarize(b_d = sum(rating - mu_hat - b_i - b_u)/(n()+lambda))
b_g_reg <- train %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_d_reg, by = 'week') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u - b_d)/(n()+lambda))
full_reg_pred <- test %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_d_reg, by = 'week') %>%
  left_join(b_g_reg, by = 'genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_d + b_g) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Regularized film, user, time, and genre effects",
                                     RMSE = RMSE(test$rating, full_reg_pred)))
RMSE_results

##### FINAL RESULTS #####

#### Using same lambda printed and coded above, use edx to 
#### train regularized film, user, date, and genre effects
#### and calculate predictions for validation data. Calculate
#### RMSE and print all RMSE values:
lambda <- lambdas[which.min(rmse_cv)]
b_i_reg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u_reg <- edx %>%
  left_join(b_i_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
b_d_reg <- edx %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  group_by(week) %>%
  summarize(b_d = sum(rating - mu_hat - b_i - b_u)/(n()+lambda))
b_g_reg <- edx %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_d_reg, by = 'week') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u - b_d)/(n()+lambda))
final_pred <- validation %>%
  left_join(b_i_reg, by = 'movieId') %>%
  left_join(b_u_reg, by = 'userId') %>%
  left_join(b_d_reg, by = 'week') %>%
  left_join(b_g_reg, by = 'genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_d + b_g) %>%
  .$pred

RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method = "Final model",
                                     RMSE = RMSE(validation$rating, final_pred)))
RMSE_results