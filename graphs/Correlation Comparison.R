library(ggplot2)

path <- dirname(rstudioapi::getSourceEditorContext()$path)

DemonAttack <- paste(path, "/demonattack.csv", sep = "")
Carnival <- paste(path, "/carnival.csv", sep = "")
AirRaid <- paste(path, "/airraid.csv", sep = "")
SpaceInvaders <- paste(path, "/spaceinvaders.csv", sep = "")


file_names = list(DemonAttack, Carnival, AirRaid, SpaceInvaders)

data <- lapply(file_names, function(file) {
  temp_data <- read.csv(file, header = TRUE)
  temp_data$Game <- gsub(".csv", "", basename(file))  # Extract game name from file path
  return(temp_data)
})

# Combine data into a single data frame
data <- do.call(rbind, data)

# Plot using ggplot
ggplot(data, aes(x = Update, y = Score, color = Game)) +
  geom_line(size = 1, alpha = 0.8) +  # Increase line thickness and add transparency
  labs(x = "Update", y = "Score", color = "Game") +
  ggtitle("Scores Over Updates for Multiple Games") +
  scale_color_discrete(labels = c("DemonAttack", "Carnival", "AirRaid", "None")) +  # Set legend labels
  theme_minimal() +
  theme(legend.position = "bottom")  # Adjust legend position



#ggplot(DemonAttack, aes(x=Update, y=Score)) + ggtitle("Relation of Average Score and Number of Network Updates") +
#  xlab("Number of Network Updates") + ylab("Average Score over 50 Games") + geom_line() + geom_point()


#plot(DemonAttack,type = "o",col = "red", xlab = "Number of Network Updates", ylab = "Average points over 50 games", 
#     main = "Progression of Agents")

#lines(Carnival, type = "o", col = "blue")
#lines(AirRaid, type = "o", col = "darkgreen")
#lines(SpaceInvaders, type = "o", col = "black")
