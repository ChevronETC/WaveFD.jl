using Printf, Statistics

function export_markdown_mcells(filename, results)
    nthreads = results.benchmarkgroup["2DAcoIsoDenQ_DEO2_FDTD"].tags[1]["nthreads"]
    μ = zeros(6, length(nthreads))
    σ = zeros(6, length(nthreads))

    rows = ["2DAcoIsoDenQ_DEO2_FDTD", "2DAcoVTIDenQ_DEO2_FDTD", "2DAcoTTIDenQ_DEO2_FDTD",
        "3DAcoIsoDenQ_DEO2_FDTD", "3DAcoVTIDenQ_DEO2_FDTD", "3DAcoTTIDenQ_DEO2_FDTD"]
    columns = ["$i threads" for i in nthreads]

    for (iprop,prop) in enumerate(rows)
        ncells = results.benchmarkgroup[prop].tags[1]["ncells"]
        for (ithread,nthreads) in enumerate(columns)
            benchmark = results.benchmarkgroup[prop][nthreads]
            x = (ncells / 1_000_000) ./ (benchmark.times .* 1e-9) # Mega-Cells per second
            μ[iprop,ithread] = mean(x)
            σ[iprop,ithread] = std(x)
        end
    end

    io = open(filename, "w")

    cpuname = Sys.cpu_info()[1].model
    write(io, @sprintf("# WaveFD Propagator Throughput\n"))
    write(io, @sprintf("## %s\n", cpuname))

    write(io, "|    ")
    for column in columns
        write(io, " | $column")
    end
    write(io, "|\n")
    write(io, "|------")
    for column in columns
        write(io, "| ------ ")
    end
    write(io, "|\n")
    for (irow,row) in enumerate(rows)
        write(io, "| $row")
        for icol = 1:length(columns)
            _μ = @sprintf("%.2f", μ[irow,icol])
            _σ = @sprintf("%.2f", 100* (σ[irow,icol] / μ[irow,icol]))
            write(io, " | $_μ MC/s ($_σ %)")
        end
        write(io, "|\n")
    end
    close(io)
end
