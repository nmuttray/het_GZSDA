#data extraction from https://github.com/passt/miceandmen/tree/master/R
library(Seurat)

load(file='mouseBMMNC-Seurat.RData')
mBMMNC <- UpdateSeuratObject(mBMMNC)
###########################
mAnnot.res1.1 <- read.table('label_mice.txt', sep='\t', header=T)
mIdent <- factor(mBMMNC@meta.data$res.1.1)#, ordered = T)
levels(mIdent) <- mAnnot.res1.1$CellType
x.mouse_raw <- t(as.matrix(GetAssayData(object = mBMMNC, slot = "counts")))
mouse_raw <- data.frame(x.mouse_raw)
mouse_raw$label <- mIdent
write.csv(mouse_raw, 'bm_mouse_raw.csv', row.names = T)

# human Seurat object
load(file='humanBMMNC-Seurat.RData')
hBMMNC <- UpdateSeuratObject(hBMMNC)
# human manual cluster annotation
hAnnot.res1.1 <- read.table('label_human.txt', sep='\t', header=T)
hIdent <- factor(hBMMNC@meta.data$res.1.1)#, ordered = T)
levels(hIdent) <- hAnnot.res1.1$CellType[1:16]
x.human_raw <- t(as.matrix(GetAssayData(object = hBMMNC, slot = "counts")))
human_raw <- data.frame(x.human_raw)
human_raw$label <- hIdent
write.csv(human_raw, 'bm_human_raw.csv', row.names = T)