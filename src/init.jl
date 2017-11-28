using Wavelets
using TimeWarp
using TimeWarp.WarpPlots
using MDCT
using APL

### SETUP BEGIN ###
l = 25000
u = zeros(l)
v = zeros(l)

for i = 2:l
	u[i] = u[i-1]+(randn()*0.01)
	v[i] = v[i-1]+(randn()*0.01)
end

v += 1+(circshift(-u, 100)+(randn(l)*0.01))
u .+= 1

sv = AbstractArray{Float64, 1}(v[1:12500])
su = u[1:12500]

tv = v[12501:end]
tu = u[12501:end]
### SETUP END ###

### MDCT BEGIN ###
dv = mdct(sv)
du = mdct(su)
### MDCT END ###

### OMPA BEGIN ###
function mm(c::Array{Float64, 1}, il::Int64, si::Int64)
	cl = length(c)
	r = zeros(cl, il)

	for i = 1:cl
		for j = 1:il

			r[i, j] = (1/cl)*c[i]*cos((pi/cl)*(j-0.5+(cl/2))*(si+i+0.5))

		end
	end

	return r
end

p = 0

mv = mm(dv, 6250, 0+p)
mu = mm(du, 6250, 0+p)

rv = matchingpursuit(sv[(1+p):(6250+p)], (x)->mv*x, (x)->mv'*x, 1)
ru = matchingpursuit(su[(1+p):(6250+p)], (x)->mu*x, (x)->mu'*x, 1)
### OMPA END ###

### DTW BEGIN ###
plot([fastdtw(rv, ru, radius)[1] for radius=1:100, i=1:10 ], show=true)
### DTW END ###
