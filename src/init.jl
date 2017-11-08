using Plots
using APL

v = zeros(40)
n = 1000000
j = 1
d = 0

v = v/(n+d)

plot(v)
savefig("p_p.png")
