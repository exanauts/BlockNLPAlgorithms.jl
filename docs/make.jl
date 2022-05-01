using Documenter
using BlockNLPAlgorithms

makedocs(
    sitename = "BlockNLPAlgorithms",
    format = Documenter.HTML(),
    modules = [BlockNLPAlgorithms],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
