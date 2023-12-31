library(nlshrink)
library(RcppCNPy)
new_dir <- paste(getwd(), "/rscripts", sep="")
setwd(new_dir)
#library(reticulate)
#np <- import("numpy")
#args = commandArgs(trailingOnly=TRUE)
#ret <- np$load(args[1])

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("NLS_in_%s.npy", uid)
X <- npyLoad(inp, dotranspose=FALSE)
print("Loaded X")
NLS_cov <- nlshrink_cov(X, k=1)
print("Finished with NLS")
out = sprintf("NLS_out_%s.npy", uid)
npySave(out, NLS_cov)
