library(ggplot2)

path <- dirname(rstudioapi::getSourceEditorContext()$path)

DemonAttack <- read.csv(paste(path, "/Demon1k.csv", sep = ""))$Score
Carnival <- read.csv(paste(path, "/Carnaval1k.csv", sep = ""))$Score
AirRaid <- read.csv(paste(path, "/Airraid1k.csv", sep = ""))$Score
SpaceInvaders <- read.csv(paste(path, "/Space1k.csv", sep = ""))$Score

kruskal_test <- kruskal.test(list(DemonAttack, Carnival, AirRaid, SpaceInvaders))

DVC <- wilcox.test(DemonAttack, Carnival, alternative = "greater")
DVA <- wilcox.test(DemonAttack, AirRaid, alternative = "greater")
DVS <- wilcox.test(DemonAttack, SpaceInvaders, alternative = "greater")

CVA <- wilcox.test(Carnival, AirRaid, alternative = "greater")
CVS <- wilcox.test(Carnival, SpaceInvaders, alternative = "greater")

AVS <- wilcox.test(AirRaid, SpaceInvaders, alternative = "greater")


# Print the result
print(DVC)