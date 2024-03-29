library(ggplot2)

path <- dirname(rstudioapi::getSourceEditorContext()$path)

DemonAttack <- paste(path, "/demonattack.csv", sep = "")
Carnival <- paste(path, "/carnival.csv", sep = "")
AirRaid <- paste(path, "/airraid.csv", sep = "")
SpaceInvaders <- paste(path, "/spaceinvaders.csv", sep = "")


file_names = list(DemonAttack, Carnival, AirRaid, SpaceInvaders)

data <- lapply(file_names, function(file) {
  temp_data <- read.csv(file, header = TRUE)
  temp_data$Game <- gsub(".csv", "", basename(file))
  return(temp_data)
})


data <- do.call(rbind, data)

ggplot(data, aes(x = Update, y = Score, color = Game)) +
  geom_line(size = 1, alpha = 0.8) +
  labs(x = "Number of Network Updates", y = "Average Score over 50 games", color = "Previously Trained On") +
  ggtitle("Performance in Space Invaders") +
  scale_color_discrete(labels = c("DemonAttack", "Carnival", "AirRaid", "None")) +
  theme_minimal() +
  theme(legend.position = "bottom")
