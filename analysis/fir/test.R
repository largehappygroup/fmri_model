# R code for brains below these graphs
install.packages("remotes")
remotes::install_github("LCBC-UiO/ggsegHO")
library(ggseg)

ggseg(atlas = hoCort, mapping = aes(fill = region)) +
  scale_fill_brain("hoCort", package = "ggsegHO") +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 6)) +
  guides(fill = guide_legend(ncol = 2))
