BUILD_HOME = pwd()

#
# clean-up after old builds
#
for dir in ("downloads", "src", "usr", "usr/lib", "usr/include")
    try
        run(`rm -rf $(BUILD_HOME)/$(dir)`)
        run(`mkdir -p $(BUILD_HOME)/$(dir)`)
    catch
        @warn "Unable to fully clean-up from previous build, likely due to nfs"
    end
end

#
# build
#

# Wave.jl
cd("$(BUILD_HOME)/../src")
run(`make`)
run(`make install`)
run(`make clean`)
cd("$(BUILD_HOME)")
