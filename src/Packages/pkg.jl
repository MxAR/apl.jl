@everywhere module pkg
	##===================================================================================
	##  save package list
	##===================================================================================
	export save_pkg_list

	##-----------------------------------------------------------------------------------
	function save_pkg_list()
		p = *(homedir(), "/.julia/pkg_list")
		isfile(p) && rm(p; force=true)
		IO = open(p, true, true, true, false, false)
		write(IO, join(collect(keys(Pkg.installed())), "+"))
		close(IO)
	end


	##===================================================================================
	##  install all packages listed in the package list
	##===================================================================================
	export install_pkg_from_list

	##-----------------------------------------------------------------------------------
	function install_pkg_from_list()
		p = *(homedir(), "/.julia/pkg_list"); @assert isfile(p)
		IO = open(p, true, false, false, false, false)
		d = split(readstring(IO), "+")
		close(IO)

		Pkg.update()
		for n in d
			Pkg.add(n)
		end
	end
end
