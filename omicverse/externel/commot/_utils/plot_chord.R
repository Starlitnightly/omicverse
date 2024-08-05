library("circlize")

data.frame <- read.csv(file = 'plot_chord_data.csv')
color.data <- read.table(file = 'plot_chord_cmap.csv', sep=',', comment.char='')
grid.col <- as.character(color.data[["V2"]])
names(grid.col) <- as.character(color.data[["V1"]])
chordDiagram(data.frame, self.link = 2,grid.col=grid.col, directional=1, scale = FALSE, link.arr.type = "big.arrow", direction.type = c("diffHeight", "arrows"), annotationTrack = c("name","grid"))
