using Documenter, WaveFD

makedocs(sitename = "WaveFD", modules=[WaveFD])

deploydocs(
    repo = "github.com/ChevronETC/WaveFD.jl.git",
)
